import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{Word2Vec, Tokenizer, StopWordsRemover}
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.hadoop.fs.{FileSystem, Path}

object Word2VecEmbed2 {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("BDA-Word2Vec-Embed")
      .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
      .config("spark.executor.memory", "4g")
      .config("spark.driver.memory", "4g")
      .config("spark.sql.shuffle.partitions", "40") // safer default
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.skewJoin.enabled", "true")
      .getOrCreate()

    import spark.implicits._

    println("=== LOADING ITEMS FROM HDFS ===")

    val rawItems = spark.read
      .parquet("hdfs://namenode:8020/bda/processed/bert_items/")
      .filter(col("combinedText").isNotNull)
      .filter(length(col("combinedText")) > 50)
      .select("asin", "combinedText", "avgRating", "reviewCount")

    println("Filtering items with reviewCount >= 5 ...")

    val topItems = rawItems
      .filter(col("reviewCount") >= 5)
      .repartition(60)
      .cache()

    println(s"Items after filter: ${topItems.count()}")

    // ─────────────────────────────
    // Tokenization
    // ─────────────────────────────
    println("Tokenizing text...")
    val tokenizer = new Tokenizer()
      .setInputCol("combinedText")
      .setOutputCol("rawTokens")

    val tokenized = tokenizer.transform(topItems)

    // ─────────────────────────────
    // Stopword removal
    // ─────────────────────────────
    println("Removing stop words...")
    val remover = new StopWordsRemover()
      .setInputCol("rawTokens")
      .setOutputCol("tokens")

    val cleaned = remover.transform(tokenized)
      .filter(size(col("tokens")) > 3)
      .select("asin", "tokens", "avgRating", "reviewCount")
      .repartition(spark.sparkContext.defaultParallelism)
      .cache()

    println(s"Items after cleaning: ${cleaned.count()}")

    // ─────────────────────────────
    // Model Path
    // ─────────────────────────────
    val modelPath = "hdfs://namenode:8020/bda/models/word2vec_model/"
    val embeddingPath = "hdfs://namenode:8020/bda/models/word2vec_embeddings/"

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val modelExists = fs.exists(new Path(modelPath))

    // ─────────────────────────────
    // Train OR Load Word2Vec
    // ─────────────────────────────
    val w2vModel =
      if (modelExists) {
        println("Loading existing Word2Vec model from HDFS...")
        Word2VecModel.load(modelPath)
      } else {
        println("Training new Word2Vec model...")

        val word2vec = new Word2Vec()
          .setInputCol("tokens")
          .setOutputCol("embedding")
          .setVectorSize(50)  // memory-efficient
          .setMinCount(50)    // controls vocabulary explosion
          .setMaxIter(5)      // slightly increased for stability
          .setNumPartitions(spark.sparkContext.defaultParallelism)
          .setWindowSize(5)
          .setStepSize(0.025)

        val model = word2vec.fit(cleaned)

        println("Saving Word2Vec model to HDFS...")
        model.write.overwrite().save(modelPath)

        model
      }

    // ─────────────────────────────
    // Generate Item Embeddings
    // ─────────────────────────────
    println("Generating item embeddings...")

    val itemEmbeddings = w2vModel.transform(cleaned)
      .select("asin", "embedding", "avgRating", "reviewCount")
      .repartition(spark.sparkContext.defaultParallelism)
      .cache()

    println(s"Items with embeddings: ${itemEmbeddings.count()}")

    // ─────────────────────────────
    // Optional Similarity Test
    // ─────────────────────────────
    println("=== SAMPLE SIMILARITY TEST ===")

    Seq("battery", "camera", "wireless").foreach { word =>
      try {
        println(s"Words similar to '$word':")
        w2vModel.findSynonyms(word, 5).show(truncate = false)
      } catch {
        case _: Exception =>
          println(s"'$word' not found in vocabulary.")
      }
    }

    // ─────────────────────────────
    // Save Embeddings to HDFS
    // ─────────────────────────────
    println("Saving item embeddings to HDFS...")

    itemEmbeddings.write
      .mode("overwrite")
      .parquet(embeddingPath)

    println("Embeddings saved successfully.")

    // ─────────────────────────────
    // Cleanup
    // ─────────────────────────────
    topItems.unpersist()
    cleaned.unpersist()
    itemEmbeddings.unpersist()

    println("=== WORD2VEC PIPELINE COMPLETE ===")
    spark.stop()
  }
}