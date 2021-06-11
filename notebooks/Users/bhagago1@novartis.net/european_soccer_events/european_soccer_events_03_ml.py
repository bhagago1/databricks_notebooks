# Databricks notebook source
# MAGIC %md ## European Soccer Events Analysis: Machine Learning
# MAGIC 
# MAGIC In this notebook, we use [Gradient-boosted tree](https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#gradient-boosted-tree-classifier) classifier to fit a model on transformed soccer events data, that could help predict whether a combination of on-field conditions lead to a goal or not.

# COMMAND ----------

# MAGIC %sql USE EURO_SOCCER_DB

# COMMAND ----------

# DBTITLE 1,Explore data to establish features
# MAGIC %sql SELECT * FROM GAME_EVENTS

# COMMAND ----------

# DBTITLE 1,Create dataset for model training and prediction
gameEventsDf = spark.sql("select event_type_str, event_team, shot_place_str, location_str, assist_method_str, situation_str, country_code, is_goal from game_events")

# COMMAND ----------

# DBTITLE 1,Necessary imports for Spark ML pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

# DBTITLE 1,Create a list of categorical features
categFeatures = ["event_type_str", "event_team", "shot_place_str", "location_str", "assist_method_str", "situation_str", "country_code"]

# COMMAND ----------

# DBTITLE 1,Encode categorical string cols to label indices
stringIndexers = [StringIndexer().setInputCol(baseFeature).setOutputCol(baseFeature + "_idx") for baseFeature in categFeatures]

# COMMAND ----------

# DBTITLE 1,Convert categorical label indices to binary vectors
encoders = [OneHotEncoder().setInputCol(baseFeature + "_idx").setOutputCol(baseFeature + "_vec") for baseFeature in categFeatures]

# COMMAND ----------

# DBTITLE 1,Combine all columns into a single feature vector
featureAssembler = VectorAssembler()
featureAssembler.setInputCols([baseFeature + "_vec" for baseFeature in categFeatures])
featureAssembler.setOutputCol("features")

# COMMAND ----------

# DBTITLE 1,Create Spark ML pipeline using a GBT classifier
gbtClassifier = GBTClassifier(labelCol="is_goal", featuresCol="features", maxDepth=5, maxIter=20)

pipelineStages = stringIndexers + encoders + [featureAssembler, gbtClassifier]
pipeline = Pipeline(stages=pipelineStages)

# COMMAND ----------

# DBTITLE 1,Split dataset into training/test, and create a model from training data
(trainingData, testData) = gameEventsDf.randomSplit([0.75, 0.25])
model = pipeline.fit(trainingData)

# COMMAND ----------

# DBTITLE 1,Validate the model on test data, display predictions
predictions = model.transform(testData)
display(predictions.select("prediction", "is_goal", "features"))

# COMMAND ----------

# DBTITLE 1,Evaluate the model using areaUnderROC metric
evaluator = BinaryClassificationEvaluator(
    labelCol="is_goal", rawPredictionCol="prediction")
evaluator.evaluate(predictions)