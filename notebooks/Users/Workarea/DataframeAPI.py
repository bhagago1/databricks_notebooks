# Databricks notebook source
spark

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/structured-streaming/events/

# COMMAND ----------

from pyspark.sql.types import *

inputPath = "/databricks-datasets/structured-streaming/events/"

# Since we know the data format already, let's define the schema to speed up processing (no need for Spark to infer schema)
jsonSchema = StructType([ StructField("time", TimestampType(), True), StructField("action", StringType(), True) ])

# Static DataFrame representing data in the JSON files
staticInputDF = (
  spark
    .read
    .schema(jsonSchema)
    .json(inputPath)
)
# Register the DataFrame as table 'static_input'
staticCountsDF.createOrReplaceTempView("static_input")

display(staticInputDF)

# COMMAND ----------

# MAGIC %sql Select * from static_input

# COMMAND ----------

from pyspark.sql.functions import *      # for window() function

staticCountsDF = (
  staticInputDF
    .groupBy(
       staticInputDF.action, 
       window(staticInputDF.time, "1 hour"))    
    .count()
)
staticCountsDF.cache()

# Register the DataFrame as table 'static_counts'
staticCountsDF.createOrReplaceTempView("static_counts")

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from static_counts order by time, action

# COMMAND ----------

from pyspark.sql.functions import *

# Similar to definition of staticInputDF above, just using `readStream` instead of `read`
streamingInputDF = (
  spark
    .readStream                       
    .schema(jsonSchema)               # Set the schema of the JSON data
    .option("maxFilesPerTrigger", 1)  # Treat a sequence of files as a stream by picking one file at a time
    .json(inputPath)
)

# Same query as staticInputDF
streamingCountsDF = (                 
  streamingInputDF
    .groupBy(
      streamingInputDF.action, 
      window(streamingInputDF.time, "1 hour"))
    .count()
)

# Is this DF actually a streaming DF?
streamingCountsDF.isStreaming

# COMMAND ----------

spark.conf.set("spark.sql.shuffle.partitions", "2")  # keep the size of shuffles small

query = (
  streamingCountsDF
    .writeStream
    .format("memory")        # memory = store in-memory table (for testing only in Spark 2.0)
    .queryName("counts")     # counts = name of the in-memory table
    .outputMode("complete")  # complete = all the counts should be in the table
    .start()
)

# COMMAND ----------

from time import sleep
sleep(5)  # wait a bit for computation to start

# COMMAND ----------

sleep(5)  # wait a bit more for more data to be computed

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from counts order by time, action

# COMMAND ----------

sleep(5)  # wait a bit more for more data to be computed

# COMMAND ----------

# MAGIC %sql select action, date_format(window.end, "MMM-dd HH:mm") as time, count from counts order by time, action

# COMMAND ----------

# MAGIC %sql select action, sum(count) as total_count from counts group by action order by action

# COMMAND ----------

