import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object Preprocess {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("BDA-Preprocess")
      .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")
      .config("spark.executor.memory", "4g")
      .config("spark.driver.memory", "2g")
      .config("spark.sql.shuffle.partitions", "100")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size", "1g")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setCheckpointDir(
      "hdfs://namenode:8020/bda/checkpoints")

    println("=== LOADING RAW DATA FROM HDFS ===")

    val schema = new StructType()
      .add("overall",        DoubleType,  nullable = true)
      .add("verified",       BooleanType, nullable = true)
      .add("reviewTime",     StringType,  nullable = true)
      .add("reviewerID",     StringType,  nullable = true)
      .add("asin",           StringType,  nullable = true)
      .add("reviewerName",   StringType,  nullable = true)
      .add("reviewText",     StringType,  nullable = true)
      .add("summary",        StringType,  nullable = true)
      .add("unixReviewTime", LongType,    nullable = true)
      .add("vote",           StringType,  nullable = true)
      .add("style",          MapType(StringType, StringType), nullable = true)
      .add("image",          ArrayType(StringType), nullable = true)

    val rawDF = spark.read
      .schema(schema)
      .json("hdfs://namenode:8020/bda/raw/Electronics.json.gz")

    val rawCount = rawDF.count()
    println(s"Raw row count: $rawCount")

    // ── Step 1: Drop rows with missing critical fields ──
    val cleanDF = rawDF
      .filter(col("reviewerID").isNotNull)
      .filter(col("asin").isNotNull)
      .filter(col("overall").isNotNull)
      .filter(col("overall") >= 1.0 && col("overall") <= 5.0)
      .drop("vote", "style", "image", "reviewTime", "reviewerName")
      .cache()

    val cleanCount = cleanDF.count()
    println(s"After cleaning: $cleanCount")

    // ── Step 2: Build user and item index ──
    println("Building user index...")
    val userIndexRDD = cleanDF
      .select("reviewerID")
      .distinct()
      .rdd
      .map(row => row.getString(0))
      .zipWithIndex()

    val userIndex = userIndexRDD
      .map { case (reviewerID, idx) => (reviewerID, idx) }
      .toDF("reviewerID", "userIndex")
      .cache()

    println("Building item index...")
    val itemIndexRDD = cleanDF
      .select("asin")
      .distinct()
      .rdd
      .map(row => row.getString(0))
      .zipWithIndex()

    val itemIndex = itemIndexRDD
      .map { case (asin, idx) => (asin, idx) }
      .toDF("asin", "itemIndex")
      .cache()

    println(s"Total users: ${userIndex.count()}")
    println(s"Total items: ${itemIndex.count()}")

    // ── Step 3: Build ALS dataset ──
    println("Building ALS dataset...")
    val alsDF = cleanDF
      .join(userIndex, "reviewerID")
      .join(itemIndex, "asin")
      .select(
        col("userIndex").cast(IntegerType).alias("userId"),
        col("itemIndex").cast(IntegerType).alias("itemId"),
        col("overall").cast(FloatType).alias("rating"),
        col("unixReviewTime").alias("timestamp")
      )

    val alsCount = alsDF.count()
    println(s"ALS dataset rows: $alsCount")

    // ── Step 4: Save ALS dataset to HDFS as Parquet ──
    println("Saving ALS dataset to HDFS...")
    alsDF
      .repartition(100)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/processed/als_ratings/")
    println("ALS dataset saved.")

    // ── Step 5: Build BERT dataset ──
    println("Building BERT dataset...")
    val bertDF = cleanDF
      .filter(col("reviewText").isNotNull || col("summary").isNotNull)
      .withColumn("text",
        when(col("reviewText").isNotNull, col("reviewText"))
          .otherwise(col("summary"))
      )
      .groupBy("asin")
      .agg(
        concat_ws(" ", collect_list("text")).alias("combinedText"),
        count("*").alias("reviewCount"),
        avg("overall").alias("avgRating")
      )
      .filter(length(col("combinedText")) > 10)

    val bertCount = bertDF.count()
    println(s"BERT dataset items: $bertCount")

    println("Saving BERT dataset to HDFS...")
    bertDF
      .repartition(50)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/processed/bert_items/")
    println("BERT dataset saved.")

    // ── Step 6: Save ID mappings ──
    println("Saving ID mappings...")
    userIndex
      .repartition(10)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/processed/user_index/")

    itemIndex
      .repartition(10)
      .write
      .mode("overwrite")
      .parquet("hdfs://namenode:8020/bda/processed/item_index/")

    println("ID mappings saved.")
    println("=== PREPROCESSING COMPLETE ===")

    spark.stop()
  }
}