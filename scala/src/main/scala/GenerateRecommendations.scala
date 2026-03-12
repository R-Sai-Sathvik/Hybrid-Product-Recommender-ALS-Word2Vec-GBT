import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.expressions.Window

object GenerateRecommendations {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("GenerateRecommendations")
      .master("spark://spark-master:7077")
      .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
      .config("spark.driver.memory", "4g")
      .config("spark.executor.memory", "4g")
      .config("spark.sql.shuffle.partitions", "40")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.skewJoin.enabled", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("\n========================================")
    println("   STEP 5: Generate Recommendations V3")
    println("========================================\n")

    // ─────────────────────────────────────────────────
    // UDFs
    // ─────────────────────────────────────────────────

    // Vector magnitude
    val vecNorm = udf((v: Vector) =>
      math.sqrt(v.toArray.map(x => x * x).sum)
    )

    // Cosine similarity between two vectors
    val cosineSim = udf((a: Vector, b: Vector) => {
      val dot   = a.toArray.zip(b.toArray).map { case (x, y) => x * y }.sum
      val normA = math.sqrt(a.toArray.map(x => x * x).sum)
      val normB = math.sqrt(b.toArray.map(x => x * x).sum)
      if (normA == 0.0 || normB == 0.0) 0.0
      else dot / (normA * normB)
    })

    // Weighted average embedding — builds unique taste profile per user
    // Weights each item's embedding by the rating the user gave it
    val weightedEmbedding = udf((embeddings: Seq[Vector], ratings: Seq[Double]) => {
      if (embeddings == null || embeddings.isEmpty) null
      else {
        val size        = embeddings.head.size
        val totalWeight = ratings.sum
        val result      = Array.fill(size)(0.0)
        embeddings.zip(ratings).foreach { case (emb, r) =>
          emb.toArray.zipWithIndex.foreach { case (v, i) =>
            result(i) += v * r
          }
        }
        if (totalWeight > 0) Vectors.dense(result.map(_ / totalWeight))
        else Vectors.dense(result)
      }
    })

    // Extract GBT probability of class 1
    val extractProb = udf((v: Vector) => v(1))

    // ─────────────────────────────────────────────────
    // STEP 1: Load saved models (NO retraining)
    // ─────────────────────────────────────────────────
    println(">>> STEP 1: Loading saved models...")
    val alsModel = ALSModel.load("hdfs://namenode:8020/bda/models/als_model")
    val gbtModel = GBTClassificationModel
      .load("hdfs://namenode:8020/bda/models/gbt_ranker_model")
    println("  ALS model loaded.")
    println("  GBT model loaded.\n")

    // ─────────────────────────────────────────────────
    // STEP 2: Load all data
    // ─────────────────────────────────────────────────
    println(">>> STEP 2: Loading data...")

    val ratings = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/als_ratings")
      .cache()

    val itemIndex = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/item_index")
      .withColumnRenamed("itemIndex", "itemId")

    val w2vWithId = spark.read
      .parquet("hdfs://namenode:8020/bda/models/word2vec_embeddings")
      .join(itemIndex, "asin")
      .select(
        col("itemId").cast("integer"),
        col("embedding"),
        col("avgRating").alias("itemAvgRating"),
        col("reviewCount").alias("itemReviewCount")
      ).cache()

    val userActivity = ratings
      .groupBy("userId")
      .agg(count("*").alias("userRatingCount"))
      .cache()

    val itemPopularity = ratings
      .groupBy("itemId")
      .agg(count("*").alias("itemRatingCount"))
      .cache()

    println(s"  Ratings      : ${ratings.count()}")
    println(s"  W2V items    : ${w2vWithId.count()}\n")

    // ─────────────────────────────────────────────────
    // STEP 3: Segment users — warm vs cold
    // ─────────────────────────────────────────────────
    println(">>> STEP 3: Segmenting 500K sampled users...")

    val sampledUsers = ratings
      .select("userId").distinct()
      .orderBy(rand(seed = 42))
      .limit(500000)
      .join(userActivity, "userId")
      .cache()

    val warmUsers = sampledUsers
      .filter(col("userRatingCount") > 5)
      .select("userId").cache()

    val coldUsers = sampledUsers
      .filter(col("userRatingCount") <= 5)
      .select("userId").cache()

    val warmCount = warmUsers.count()
    val coldCount = coldUsers.count()
    println(s"  Warm users (>5 ratings)  : $warmCount")
    println(s"  Cold users (<=5 ratings) : $coldCount\n")

    // ═════════════════════════════════════════════════
    // PATH A: Warm users → ALS + GBT (personalized)
    // ═════════════════════════════════════════════════
    println(">>> PATH A: ALS + GBT for warm users...")

    val userWindow = Window.partitionBy("userId")

    val alsRecs = alsModel.recommendForUserSubset(warmUsers, 50)
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
            (col("alsRawScore") - col("alsMin")) /
            (col("alsMax") - col("alsMin"))
          )
      )
      .select("userId", "itemId", "alsScore")

    val warmFeatures = alsNorm
      .join(w2vWithId,      Seq("itemId"), "left")
      .join(userActivity,   Seq("userId"), "left")
      .join(itemPopularity, Seq("itemId"), "left")
      .withColumn("w2vScore",
        when(col("embedding").isNull, lit(0.0))
          .otherwise(vecNorm(col("embedding")))
      )
      .withColumn("itemAvgRating",   coalesce(col("itemAvgRating"),   lit(3.0)))
      .withColumn("userRatingCount", coalesce(col("userRatingCount"), lit(1L)))
      .withColumn("itemRatingCount", coalesce(col("itemRatingCount"), lit(1L)))
      .withColumn("logItemPop",      log(col("itemRatingCount") + 1.0))
      .withColumn("logUserActivity", log(col("userRatingCount") + 1.0))
      .cache()

    val assembler = new VectorAssembler()
      .setInputCols(Array("alsScore","w2vScore","logItemPop",
                          "logUserActivity","itemAvgRating"))
      .setOutputCol("features")

    val warmRecs = gbtModel
      .transform(assembler.transform(warmFeatures))
      .withColumn("hybridScore", extractProb(col("probability")))
      .withColumn("alsScore",    col("alsScore"))
      .withColumn("w2vScore",    col("w2vScore"))
      .withColumn("source",      lit("als_gbt"))
      .select("userId","itemId","hybridScore","alsScore","w2vScore","source")
      .withColumn("rank", rank().over(
        Window.partitionBy("userId").orderBy(desc("hybridScore"))))
      .filter(col("rank") <= 10)
      .cache()

    val warmCovered = warmRecs.select("userId").distinct().count()
    println(s"  Warm users covered : $warmCovered / $warmCount\n")

    // ═════════════════════════════════════════════════
    // PATH B: Cold users → Personalized W2V
    //
    // Each cold user rated 1-5 items.
    // We find W2V embeddings of those rated items,
    // compute a WEIGHTED AVERAGE embedding (by rating) =
    // the user's unique taste profile vector.
    // Then cosine similarity vs top-1000 candidate items
    // gives each user a DIFFERENT personalised list.
    // ═════════════════════════════════════════════════
    println(">>> PATH B: Personalized W2V for cold users...")

    // Get rated items for cold users that have W2V embeddings
    val coldUserRatings = ratings
      .join(coldUsers, "userId")
      .join(
        w2vWithId.select(col("itemId"), col("embedding")),
        Seq("itemId"), "inner"
      )
      .select(
        col("userId"),
        col("itemId"),
        col("rating").cast("double"),
        col("embedding")
      )
      .cache()

    // Build weighted average taste profile per cold user
    val coldUserProfiles = coldUserRatings
      .groupBy("userId")
      .agg(
        collect_list(col("embedding")).alias("embeddings"),
        collect_list(col("rating").cast("double")).alias("ratings")
      )
      .withColumn("userProfile",
        weightedEmbedding(col("embeddings"), col("ratings"))
      )
      .filter(col("userProfile").isNotNull)
      .select("userId", "userProfile")
      .cache()

    // Cold users whose rated items had NO W2V embeddings at all
    val coldNoProfile = coldUsers
      .join(coldUserProfiles, Seq("userId"), "left_anti")
      .cache()

    val profileCount   = coldUserProfiles.count()
    val noProfileCount = coldNoProfile.count()
    println(s"  Cold users with W2V profile : $profileCount")
    println(s"  Cold users needing fallback : $noProfileCount")

    // Candidate pool: top 1000 items by popularity with W2V embeddings
    val candidatePool = w2vWithId
      .join(itemPopularity, Seq("itemId"), "left")
      .withColumn("itemRatingCount", coalesce(col("itemRatingCount"), lit(1L)))
      .filter(col("itemRatingCount") >= 10)
      .withColumn("logItemPop", log(col("itemRatingCount") + 1.0))
      .select(
        col("itemId"),
        col("embedding"),
        col("itemAvgRating"),
        col("itemRatingCount"),
        col("logItemPop")
      )
      .orderBy(desc("itemRatingCount"))
      .limit(1000)
      .cache()

    println(s"  Candidate pool             : ${candidatePool.count()} items")

    // Cross join each user profile with 1000 candidates
    // → cosine similarity → unique ranked list per user
    // → exclude items the user already rated
    val w2vRecs = coldUserProfiles
      .crossJoin(candidatePool)
      .withColumn("simScore",    cosineSim(col("userProfile"), col("embedding")))
      .withColumn("w2vScore",    vecNorm(col("embedding")))
      // Score = taste match + rating quality + mild popularity boost
      .withColumn("hybridScore",
        col("simScore")                         * 0.60 +
        (col("itemAvgRating") / lit(5.0))       * 0.25 +
        (col("logItemPop")    / lit(10.0))      * 0.15
      )
      .withColumn("alsScore", lit(0.0))
      .withColumn("source",   lit("w2v_personalized"))
      .select("userId","itemId","hybridScore","alsScore","w2vScore","source")
      // Remove items the user already rated
      .join(
        coldUserRatings.select("userId","itemId"),
        Seq("userId","itemId"), "left_anti"
      )
      .withColumn("rank", rank().over(
        Window.partitionBy("userId").orderBy(desc("hybridScore"))))
      .filter(col("rank") <= 10)
      .cache()

    val w2vCovered = w2vRecs.select("userId").distinct().count()
    println(s"  W2V personalized covered   : $w2vCovered\n")

    // ═════════════════════════════════════════════════
    // PATH C: Popularity fallback
    // Only for cold users with NO W2V embedded rated items
    // (these users rated only very obscure/unreviewed items)
    // ═════════════════════════════════════════════════
    println(">>> PATH C: Popularity fallback for remaining cold users...")

    val popularityTop10 = itemPopularity
      .join(w2vWithId, Seq("itemId"), "left")
      .withColumn("w2vScore",
        when(col("embedding").isNull, lit(0.0))
          .otherwise(vecNorm(col("embedding")))
      )
      .withColumn("itemAvgRating", coalesce(col("itemAvgRating"), lit(3.0)))
      .withColumn("logItemPop",    log(col("itemRatingCount") + 1.0))
      .withColumn("hybridScore",
        col("logItemPop") * 0.5 +
        col("itemAvgRating") * 0.3 +
        col("w2vScore") * 0.2
      )
      .orderBy(desc("hybridScore"))
      .limit(10)
      .select("itemId", "hybridScore", "w2vScore")
      .cache()

    val fallbackRecs = coldNoProfile
      .crossJoin(popularityTop10)
      .withColumn("alsScore", lit(0.0))
      .withColumn("source",   lit("popularity_fallback"))
      .select("userId","itemId","hybridScore","alsScore","w2vScore","source")
      .withColumn("rank", rank().over(
        Window.partitionBy("userId").orderBy(desc("hybridScore"))))
      .filter(col("rank") <= 10)
      .cache()

    val fallbackCovered = fallbackRecs.select("userId").distinct().count()
    println(s"  Popularity fallback users  : $fallbackCovered\n")

    // ─────────────────────────────────────────────────
    // STEP 4: Combine all three paths
    // ─────────────────────────────────────────────────
    println(">>> STEP 4: Combining all paths...")

    val finalRecs = warmRecs
      .unionByName(w2vRecs)
      .unionByName(fallbackRecs)
      .orderBy("userId", "rank")
      .cache()

    val totalRecs     = finalRecs.count()
    val usersWithRecs = finalRecs.select("userId").distinct().count()
    val totalUsers    = ratings.select("userId").distinct().count()

    println(s"\n  Total recommendations : $totalRecs")
    println(s"  Users with recs       : $usersWithRecs")
    println(f"  Coverage of sampled   : ${usersWithRecs * 100.0 / 500000}%.1f%%")
    println(f"  Coverage of all users : ${usersWithRecs * 100.0 / totalUsers}%.1f%%\n")

    println("  Breakdown by source:")
    finalRecs.groupBy("source")
      .agg(
        count("*").alias("totalRecs"),
        countDistinct("userId").alias("usersServed")
      )
      .orderBy("source")
      .show()

    finalRecs.show(30, truncate = false)

    // ─────────────────────────────────────────────────
    // STEP 5: Personalization quality check
    // ─────────────────────────────────────────────────
    println(">>> STEP 5: Personalization check...")

    val uniqueTopItems = finalRecs
      .filter(col("rank") === 1)
      .select("itemId").distinct().count()

    println(s"  Unique items at rank-1     : $uniqueTopItems")
    println(s"  Total users with recs      : $usersWithRecs")
    println(f"  Personalization ratio      : ${uniqueTopItems * 100.0 / usersWithRecs}%.1f%%")
    println("  (100% = every user gets a unique top recommendation)\n")

    println("  Sample of W2V personalized recs (first 5 users):")
    finalRecs
      .filter(col("source") === "w2v_personalized")
      .filter(col("rank") <= 3)
      .limit(15)
      .show(truncate = false)

    // ─────────────────────────────────────────────────
    // STEP 6: Save
    // ─────────────────────────────────────────────────
    println(">>> STEP 6: Saving to HDFS...")
    finalRecs.write.mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/models/hybrid_recommendations")
    println("  Saved -> /bda/models/hybrid_recommendations")

    println("\n========================================")
    println("   STEP 5 COMPLETE ✅")
    println("========================================\n")

    spark.stop()
  }
}