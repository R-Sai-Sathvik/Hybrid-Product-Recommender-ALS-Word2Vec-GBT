import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object ALSTrainer {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("BDA-ALS-Trainer")
      .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
      .config("spark.executor.memory", "5g")
      .config("spark.driver.memory", "2g")
      .config("spark.sql.shuffle.partitions", "50")
      .config("spark.default.parallelism", "6")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size", "2g")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.network.timeout", "800s")
      .config("spark.executor.heartbeatInterval", "60s")
      .config("spark.locality.wait", "3s")
      .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setCheckpointDir(
      "hdfs://namenode:8020/bda/checkpoints")

    println("=== LOADING PREPROCESSED ALS DATA ===")

    val rawRatings = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/als_ratings/")
      .select(
        col("userId").cast(IntegerType),
        col("itemId").cast(IntegerType),
        col("rating").cast(FloatType)
      )
      .na.drop()

    val totalRaw = rawRatings.count()
    println(s"Total raw ratings: $totalRaw")

    // ── KEY FIX: Filter active users only ──
    // Users with fewer than 5 ratings are useless for collaborative filtering
    // This reduces 20M users to ~2-3M active users
    // Cuts memory usage by 80% and speeds up training 5-10x
    println("Filtering to active users (minimum 5 ratings)...")

    val userRatingCounts = rawRatings
      .groupBy("userId")
      .agg(count("*").alias("ratingCount"))
      .filter(col("ratingCount") >= 5)
      .select("userId")

    val ratingsDF = rawRatings
      .join(userRatingCounts, "userId")
      .select("userId", "itemId", "rating")
      .cache()

    val filteredCount = ratingsDF.count()
    println(s"Ratings after filtering active users: $filteredCount")

    val userCount = ratingsDF.select("userId").distinct().count()
    val itemCount = ratingsDF.select("itemId").distinct().count()
    println(s"Active users: $userCount")
    println(s"Unique items: $itemCount")

    // ── Train/Test Split ──
    val Array(trainDF, testDF) = ratingsDF
      .randomSplit(Array(0.8, 0.2), seed = 42)

    trainDF.cache()
    testDF.cache()

    println(s"Train size: ${trainDF.count()}")
    println(s"Test size : ${testDF.count()}")

    // ── ALS Configuration ──
    // rank=15     : enough latent factors for quality recommendations
    // maxIter=5   : sufficient convergence, 2x faster than 10
    // numBlocks=20: smaller chunks = less memory per block
    val als = new ALS()
      .setMaxIter(5)
      .setRank(15)
      .setRegParam(0.1)
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("rating")
      .setColdStartStrategy("drop")
      .setNumUserBlocks(20)
      .setNumItemBlocks(20)
      .setImplicitPrefs(false)
      .setCheckpointInterval(2)

    println("=== TRAINING ALS MODEL ===")
    println("Estimated time: 30-60 mins")

    val model = als.fit(trainDF)
    println("Training complete.")

    // ── Evaluate ──
    println("=== EVALUATING MODEL ===")
    val predictions = model
      .transform(testDF)
      .filter(col("prediction").isNotNull)
      .filter(!isnan(col("prediction")))

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"RMSE: $rmse")

    // Good RMSE range for Amazon data: 0.8 - 1.2
    if (rmse < 0.8) println("Model quality: Excellent")
    else if (rmse < 1.2) println("Model quality: Good")
    else println("Model quality: Acceptable - consider tuning")

    // ── Sample Recommendations ──
    println("=== SAMPLE RECOMMENDATIONS (Top 5 per user) ===")
    val userRecs = model.recommendForAllUsers(5)
    userRecs.show(10, truncate = false)

    println("=== SAMPLE RECOMMENDATIONS (Top 5 per item) ===")
    val itemRecs = model.recommendForAllItems(5)
    itemRecs.show(10, truncate = false)

    // ── Save Model to HDFS ──
    println("Saving ALS model to HDFS...")
    model.write.overwrite()
      .save("hdfs://namenode:8020/bda/models/als_model/")
    println("ALS model saved.")

    // ── Save Predictions to HDFS ──
    println("Saving predictions to HDFS...")
    predictions
      .repartition(50)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/output/als_predictions/")
    println("Predictions saved.")

    // ── Save User Recs to HDFS ──
    println("Saving user recommendations to HDFS...")
    userRecs
      .repartition(20)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/output/user_recommendations/")
    println("User recommendations saved.")

    println("=== ALS TRAINING COMPLETE ===")
    spark.stop()
  }
}