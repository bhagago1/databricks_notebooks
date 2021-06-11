# Databricks notebook source
# MAGIC %md
# MAGIC # European Soccer Events Analysis: ETL
# MAGIC * A soccer game or any other sport generates many events, which help solve a number of use cases across the Sports and Media & Entertainment industries:
# MAGIC   * *Like what on-field playing conditions and events (passes, positions etc.) leads to more goals/touch-downs etc.*
# MAGIC   * *Or what does the win-loss percentage looks like with different combinations of players in different on-field positions*
# MAGIC   * *Or what does a sportsperson's performance graph look like across the years/seasons and teams etc.*
# MAGIC 
# MAGIC ![](https://databricks.com/wp-content/uploads/2018/07/European-Soccer-Events-Analysis-Diagram.png)
# MAGIC 
# MAGIC 
# MAGIC * This demo uses a European Soccer Games events dataset, and demonstrates:
# MAGIC   * *End-to-end Data Engineering pipeline including data extraction, transformation and loading*
# MAGIC   * *How to answer business questions by analyzing the transformed data - using a combination of Spark SQL and Visualizations*
# MAGIC   * *Usage of Gradient-boosted tree classifier to predict events of most significance (goals in a soccer game)*

# COMMAND ----------

# MAGIC %md ##1. Data Sourcing/Extraction
# MAGIC 
# MAGIC Dataset has been downloaded from [**Kaggle**](https://www.kaggle.com/secareanualin/football-events). It provides a granular view of 9,074 games, from the biggest 5 European football (soccer) leagues: England, Spain, Germany, Italy, France, from 2011/2012 season to 2016/2017 season as of 25.01.2017. This is what the schema looks like:
# MAGIC 
# MAGIC | Column Name | Colum Description |
# MAGIC | ----------- | ----------------- |
# MAGIC | id_odsp | unique identifier of game (odsp stands from oddsportal.com) |
# MAGIC | id_event | unique identifier of event (id_odsp + sort_order) |
# MAGIC | sort_order | chronological sequence of events in a game |
# MAGIC | time | minute of the game |
# MAGIC | text | text commentary |
# MAGIC | event_type | primary event, 11 unique events |
# MAGIC | event_type2 | secondary event, 4 unique events |
# MAGIC | side | Home or Away team |
# MAGIC | event_team | team that produced the event. In case of Own goals, event team is the team that benefited from the own goal |
# MAGIC | opponent | opposing team |
# MAGIC | player | name of the player involved in main event |
# MAGIC | player2 | name of player involved in secondary event |
# MAGIC | player_in | player that came in (only applies to substitutions) |
# MAGIC | player_out | player substituted (only applies to substitutions) |
# MAGIC | shot_place | placement of the shot, 13 possible placement locations |
# MAGIC | shot_outcome | 4 possible outcomes |
# MAGIC | is_goal | binary variable if the shot resulted in a goal (own goals included) |
# MAGIC | location | location on the pitch where the event happened, 19 possible locations |
# MAGIC | bodypart | 3 body parts |
# MAGIC | assist_method | in case of an assisted shot, 5 possible assist methods |
# MAGIC | situation | 4 types |
# MAGIC 
# MAGIC To load your data into DBFS, please refer to [Databricks Guide > Importing Data](https://docs.databricks.com/user-guide/importing-data.html).

# COMMAND ----------

# DBTITLE 1,Create appropriate data folders
dbutils.fs.mkdirs("/data/eu-soccer-events")
dbutils.fs.mkdirs("/data/eu-soccer-events/input")
dbutils.fs.mkdirs("/data/eu-soccer-events/interm")

# COMMAND ----------

# DBTITLE 1,Check availability of data files
# MAGIC %fs ls /data/eu-soccer-events/input

# COMMAND ----------

# DBTITLE 1,Take a peek at game events data
# MAGIC %sh head /data/eu-soccer-events/input/events.csv

# COMMAND ----------

# DBTITLE 1,Specify the schema for game events data
from pyspark.sql.types import *

schema = (StructType().
          add("id_odsp", StringType()).add("id_event", StringType()).add("sort_order", IntegerType()).
          add("time", IntegerType()).add("text", StringType()).add("event_type", IntegerType()).
          add("event_type2", IntegerType()).add("side", IntegerType()).add("event_team", StringType()).
          add("opponent", StringType()).add("player", StringType()).add("player2", StringType()).
          add("player_in", StringType()).add("player_out", StringType()).add("shot_place", IntegerType()).
          add("shot_outcome", IntegerType()).add("is_goal", IntegerType()).add("location", IntegerType()).
          add("bodypart", IntegerType()).add("assist_method", IntegerType()).add("situation", IntegerType()).
          add("fast_break", IntegerType())
         )

# COMMAND ----------

# DBTITLE 1,Extract game events data into DataFrame and display
eventsDf = (spark.read.csv("/data/eu-soccer-events/input/events.csv", 
                         schema=schema, header=True, 
                         ignoreLeadingWhiteSpace=True, 
                         ignoreTrailingWhiteSpace=True,
                         nullValue='NA'))

eventsDf = eventsDf.na.fill({'player': 'NA', 'event_team': 'NA', 'opponent': 'NA', 
                             'event_type': 99, 'event_type2': 99, 'shot_place': 99, 
                             'shot_outcome': 99, 'location': 99, 'bodypart': 99, 
                             'assist_method': 99, 'situation': 99})
display(eventsDf)

# COMMAND ----------

# DBTITLE 1,Extract game aggregate info into dataframe and display
gameInfDf = (spark.read.csv("/data/eu-soccer-events/input/ginf.csv", 
                         inferSchema=True, header=True, 
                         ignoreLeadingWhiteSpace=True, 
                         ignoreTrailingWhiteSpace=True,
                         nullValue="NA"))
display(gameInfDf)

# COMMAND ----------

# MAGIC %md ## 2. Data Transformation
# MAGIC Convert the data to a format, such that one could gather meaningful insights from it

# COMMAND ----------

# DBTITLE 1,Define a generic dictionary/map lookup function
def mapKeyToVal(mapping):
    def mapKeyToVal_(col):
        return mapping.get(col)
    return udf(mapKeyToVal_, StringType())

# COMMAND ----------

# DBTITLE 1,Create dictionaries/maps for useful attributes
evtTypeMap = {0:'Announcement', 1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded', 99:'NA'}

evtTyp2Map = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal', 99:'NA'}

sideMap = {1:'Home', 2:'Away'}

shotPlaceMap = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner', 99:'NA'}

shotOutcomeMap = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar', 99:'NA'}

locationMap = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded', 99:'NA'}

bodyPartMap = {1:'Right foot', 2:'Left foot', 3:'Head', 99:'NA'}

assistMethodMap = {0:'None', 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball', 99:'NA'}

situationMap = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick', 99:'NA'}

countryCodeMap = {'germany':'DEU', 'france':'FRA', 'england':'GBR', 'spain':'ESP', 'italy':'ITA'}

# COMMAND ----------

# DBTITLE 1,Map country names to codes
gameInfDf = gameInfDf.withColumn("country_code", mapKeyToVal(countryCodeMap)("country"))

display(gameInfDf['id_odsp','country','country_code'])

# COMMAND ----------

# DBTITLE 1,Transform game events data using lookups and join with high-level info
eventsDf = (
             eventsDf.
             withColumn("event_type_str", mapKeyToVal(evtTypeMap)("event_type")).
             withColumn("event_type2_str", mapKeyToVal(evtTyp2Map)("event_type2")).
             withColumn("side_str", mapKeyToVal(sideMap)("side")).
             withColumn("shot_place_str", mapKeyToVal(shotPlaceMap)("shot_place")).
             withColumn("shot_outcome_str", mapKeyToVal(shotOutcomeMap)("shot_outcome")).
             withColumn("location_str", mapKeyToVal(locationMap)("location")).
             withColumn("bodypart_str", mapKeyToVal(bodyPartMap)("bodypart")).
             withColumn("assist_method_str", mapKeyToVal(assistMethodMap)("assist_method")).
             withColumn("situation_str", mapKeyToVal(situationMap)("situation"))
           )

joinedDf = (
  eventsDf.join(gameInfDf, eventsDf.id_odsp == gameInfDf.id_odsp, 'inner').
  select(eventsDf.id_odsp, eventsDf.id_event, eventsDf.sort_order, eventsDf.time, eventsDf.event_type, eventsDf.event_type_str, eventsDf.event_type2, eventsDf.event_type2_str, eventsDf.side, eventsDf.side_str, eventsDf.event_team, eventsDf.opponent, eventsDf.player, eventsDf.player2, eventsDf.player_in, eventsDf.player_out, eventsDf.shot_place, eventsDf.shot_place_str, eventsDf.shot_outcome, eventsDf.shot_outcome_str, eventsDf.is_goal, eventsDf.location, eventsDf.location_str, eventsDf.bodypart, eventsDf.bodypart_str, eventsDf.assist_method, eventsDf.assist_method_str, eventsDf.situation, eventsDf.situation_str, gameInfDf.country_code)
)

# COMMAND ----------

# DBTITLE 1,Create time bins for game events
from pyspark.ml.feature import QuantileDiscretizer

joinedDf = QuantileDiscretizer(numBuckets=10, inputCol="time", outputCol="time_bin").fit(joinedDf).transform(joinedDf)

display(joinedDf)

# COMMAND ----------

# MAGIC %md ## 3. Data Loading
# MAGIC Load the transformed data to persistent storage, so that it's query-able across notebooks and clusters

# COMMAND ----------

# DBTITLE 1,Create a Databricks/Spark database
# MAGIC %sql 
# MAGIC   CREATE DATABASE IF NOT EXISTS EURO_SOCCER_DB
# MAGIC   LOCATION "dbfs:/FileStore/databricks-abhinav/eu-soccer-events/interm"

# COMMAND ----------

# DBTITLE 1,Set the database in session
# MAGIC %sql USE EURO_SOCCER_DB

# COMMAND ----------

# DBTITLE 1,Load transformed game event data into a Databricks/Spark table
joinedDf.write.saveAsTable("GAME_EVENTS", format = "parquet", mode = "overwrite", partitionBy = "COUNTRY_CODE", path = "dbfs:/FileStore/databricks-abhinav/eu-soccer-events/interm/tr-events")

# COMMAND ----------

# DBTITLE 1,Describe the database schema
# MAGIC %sql DESCRIBE GAME_EVENTS