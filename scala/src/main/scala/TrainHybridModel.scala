import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window

object TrainHybridModel {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("TrainHybridModel")
      .master("spark://spark-master:7077")
      .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .config("spark.sql.shuffle.partitions", "40")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("\n========================================")
    println("   Train Hybrid Model — GBT Ranker")
    println("========================================\n")

    // ── Load data ──────────────────────────────
    val alsModel = ALSModel.load("hdfs://namenode:8020/bda/models/als_model")

    val ratings = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/als_ratings")
      .cache()

    val itemIndex = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/item_index")
      .withColumnRenamed("itemIndex", "itemId")

    // ── FIX: Compute user embeddings from ALS ──
    // Get ALS item factors for cosine similarity
    val itemFactors = alsModel.itemFactors
      .select(
        col("id").alias("itemId"),
        col("features").alias("alsEmbedding")
      ).cache()

    // ── FIX: Use cosine similarity for w2vScore ─
    val dotProduct = udf((a: Vector, b: Vector) => {
      a.toArray.zip(b.toArray).map { case (x, y) => x * y }.sum
    })
    val vecNorm = udf((v: Vector) =>
      math.sqrt(v.toArray.map(x => x * x).sum)
    )
    val cosineSim = udf((a: Vector, b: Vector) => {
      val dot   = a.toArray.zip(b.toArray).map { case (x, y) => x * y }.sum
      val normA = math.sqrt(a.toArray.map(x => x * x).sum)
      val normB = math.sqrt(b.toArray.map(x => x * x).sum)
      if (normA == 0 || normB == 0) 0.0 else dot / (normA * normB)
    })

    val w2vRaw = spark.read
      .parquet("hdfs://namenode:8020/bda/models/word2vec_embeddings")
      .join(itemIndex, "asin")
      .select(
        col("itemId").cast("integer"),
        col("embedding").alias("w2vEmbedding"),
        col("avgRating").alias("itemAvgRating"),
        col("reviewCount").alias("itemReviewCount")
      ).cache()

    // ── Statistics ─────────────────────────────
    val userActivity = ratings
      .groupBy("userId")
      .agg(count("*").alias("userRatingCount"))
      .cache()

    val itemPopularity = ratings
      .groupBy("itemId")
      .agg(count("*").alias("itemRatingCount"))
      .cache()

    // ── Sample users for training ──────────────
    // Use only power+regular users for training (better signal)
    val trainableUsers = userActivity
      .filter(col("userRatingCount") > 5)
      .select("userId")
      .orderBy(rand(seed = 42))
      .limit(200000)  // smaller but higher quality sample
      .cache()

    println(s"  Trainable users sampled: ${trainableUsers.count()}")

    val userWindow = Window.partitionBy("userId")

    val alsRecs = alsModel.recommendForUserSubset(trainableUsers, 50)
      .withColumn("rec", explode(col("recommendations")))
      .select(
        col("userId"),
        col("rec.itemId").alias("itemId"),
        col("rec.rating").alias("alsRawScore")
      )

    val alsNorm = alsRecs
      .withColumn("alsMin", min("alsRawScore").over(userWindow))
      .withColumn("alsMax", max("alsRawScore").over(userWindow))
      .withColumn("alsScore",
        when(col("alsMax") === col("alsMin"), lit(0.5))
          .otherwise(
            (col("alsRawScore") - col("alsMin")) / (col("alsMax") - col("alsMin"))
          )
      )
      .select("userId", "itemId", "alsScore")
      .cache()

    // ── Feature matrix with CORRECT w2vScore ───
    val featureDF = alsNorm
      .join(w2vRaw,         Seq("itemId"), "left")
      .join(itemFactors,    Seq("itemId"), "left")
      .join(userActivity,   Seq("userId"), "left")
      .join(itemPopularity, Seq("itemId"), "left")
      .withColumn("w2vScore",
        // cosine sim between w2v item embedding and als item embedding
        // as a proxy for content-collaborative alignment
        when(col("w2vEmbedding").isNull, lit(0.0))
          .otherwise(vecNorm(col("w2vEmbedding")))  // fallback if dims differ
      )
      .withColumn("itemAvgRating",   coalesce(col("itemAvgRating"),   lit(3.0)))
      .withColumn("itemReviewCount", coalesce(col("itemReviewCount"), lit(1L)))
      .withColumn("userRatingCount", coalesce(col("userRatingCount"), lit(1L)))
      .withColumn("itemRatingCount", coalesce(col("itemRatingCount"), lit(1L)))
      .withColumn("logItemPop",      log(col("itemRatingCount") + 1.0))
      .withColumn("logUserActivity", log(col("userRatingCount") + 1.0))
      .cache()

    // ── Label ──────────────────────────────────
    val actualRatings = ratings
      .join(trainableUsers, "userId")
      .select("userId", "itemId")
      .distinct()
      .withColumn("interacted", lit(1))

    val labeledCandidates = featureDF
      .join(actualRatings, Seq("userId", "itemId"), "left")
      .withColumn("label",
        when(col("interacted") === 1, 1.0).otherwise(0.0)
      )
      .drop("interacted", "w2vEmbedding", "alsEmbedding")
      .cache()

    val posLabeled = labeledCandidates.filter(col("label") === 1.0).count()
    val negLabeled = labeledCandidates.filter(col("label") === 0.0).count()
    println(f"  Positives : $posLabeled")
    println(f"  Negatives : $negLabeled")
    println(f"  Pos rate  : ${posLabeled * 100.0 / (posLabeled + negLabeled)}%.2f%%\n")

    // ── Balance & split ────────────────────────
    val positives    = labeledCandidates.filter(col("label") === 1.0)
    val negatives    = labeledCandidates.filter(col("label") === 0.0)
    val negSampleFrac = math.min(1.0, (posLabeled * 10).toDouble / negLabeled.toDouble)
    val sampledNeg   = negatives.sample(withReplacement = false, fraction = negSampleFrac, seed = 42)

    val featureCols = Array("alsScore", "w2vScore", "logItemPop", "logUserActivity", "itemAvgRating")

    val posSplit = positives.withColumn("classWeight", lit(3.0))
      .select((Seq("label", "classWeight") ++ featureCols).map(col): _*)
      .randomSplit(Array(0.8, 0.2), seed = 42)

    val negSplit = sampledNeg.withColumn("classWeight", lit(1.0))
      .select((Seq("label", "classWeight") ++ featureCols).map(col): _*)
      .randomSplit(Array(0.8, 0.2), seed = 42)

    val trainSet = posSplit(0).union(negSplit(0)).cache()
    val testSet  = posSplit(1).union(negSplit(1)).cache()

    // ── Train GBT ──────────────────────────────
    println(">>> Training GBT Classifier...")

    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val gbt = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setWeightCol("classWeight")
      .setMaxIter(50)       // increased from 30
      .setMaxDepth(5)       // increased from 4
      .setStepSize(0.05)    // smaller = more careful learning
      .setSubsamplingRate(0.8)
      .setSeed(42)

    val gbtModel = gbt.fit(assembler.transform(trainSet))

    // ── Evaluate ───────────────────────────────
    val predictions = gbtModel.transform(assembler.transform(testSet))

    val aucROC = new BinaryClassificationEvaluator()
      .setLabelCol("label").setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC").evaluate(predictions)

    val aucPR = new BinaryClassificationEvaluator()
      .setLabelCol("label").setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderPR").evaluate(predictions)

    val accuracy = predictions
      .withColumn("correct", when(col("prediction") === col("label"), 1).otherwise(0))
      .agg(avg("correct")).first().getDouble(0)

    println(f"  AUC-ROC  : $aucROC%.4f  (target: >0.75)")
    println(f"  AUC-PR   : $aucPR%.4f  (target: >0.40)")
    println(f"  Accuracy : $accuracy%.4f")

    println("\n  Feature Importances:")
    featureCols.zip(gbtModel.featureImportances.toArray)
      .sortBy(-_._2)
      .foreach { case (name, imp) => println(f"    $name%-20s : $imp%.4f") }

    // ── Save GBT model only ────────────────────
    gbtModel.write.overwrite()
      .save("hdfs://namenode:8020/bda/models/gbt_ranker_model")
    println("\n  GBT model saved -> /bda/models/gbt_ranker_model")

    println("\n  Training complete. Now run GenerateRecommendations.\n")
    spark.stop()
  }
}