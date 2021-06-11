# Databricks notebook source
# MAGIC %md ## European Soccer Events Analysis: Ad-hoc Data Analysis
# MAGIC 
# MAGIC In this notebook, we'll analyze the transformed soccer game events to answer questions like:
# MAGIC * *What's the distribution of goals by shot place (place within the goal area)?*
# MAGIC * *What's the distribution of goals by different soccer leagues?*
# MAGIC * *What are the top locations (on-field player position) for each shot place in Spanish league?*
# MAGIC * *When/What time window are most goals scored within a game in each league?*
# MAGIC 
# MAGIC We'll also take a look at how to use third-party libraries for visualizations - like creating a 3D scatter-plot to see distribution of goals by shot place and location.

# COMMAND ----------

# MAGIC %sql USE EURO_SOCCER_DB

# COMMAND ----------

# DBTITLE 1,Take a peek at all GOALS!!!
# MAGIC %sql 
# MAGIC     SELECT * FROM GAME_EVENTS 
# MAGIC     WHERE is_goal = 1

# COMMAND ----------

# MAGIC %md ### Built-in Visualizations

# COMMAND ----------

# DBTITLE 1,Distribution of goals by shot place
# MAGIC %sql
# MAGIC   SELECT CASE WHEN shot_place_str == 'NA' THEN 'Unknown' ELSE shot_place_str END shot_place, COUNT(1) AS TOT_GOALS
# MAGIC     FROM GAME_EVENTS
# MAGIC     WHERE is_goal = 1
# MAGIC    GROUP BY shot_place_str

# COMMAND ----------

# DBTITLE 1,Distribution of goals by countries/leagues
# MAGIC %sql
# MAGIC   SELECT country_code, COUNT(1) AS TOT_GOALS
# MAGIC     FROM GAME_EVENTS
# MAGIC     WHERE is_goal = 1
# MAGIC    GROUP BY country_code

# COMMAND ----------

# DBTITLE 1,Top 3 goal locations per shot placement in the Spanish League
# MAGIC %sql
# MAGIC       SELECT SHOT_PLACE_STR, LOCATION_STR, TOT_GOALS
# MAGIC         FROM (
# MAGIC           SELECT SHOT_PLACE_STR, LOCATION_STR, TOT_GOALS,
# MAGIC                 RANK() OVER (PARTITION BY SHOT_PLACE_STR ORDER BY TOT_GOALS DESC) goals_rank
# MAGIC            FROM (
# MAGIC               SELECT CASE WHEN LOCATION_STR == 'NA' THEN 'Unknown' ELSE LOCATION_STR END LOCATION_STR, 
# MAGIC                     CASE WHEN SHOT_PLACE_STR == 'NA' THEN 'Unknown' ELSE SHOT_PLACE_STR END SHOT_PLACE_STR, 
# MAGIC                     COUNT(1) AS TOT_GOALS
# MAGIC               FROM GAME_EVENTS
# MAGIC               WHERE is_goal = 1 AND COUNTRY_CODE = 'ESP'
# MAGIC              GROUP BY SHOT_PLACE_STR, LOCATION_STR
# MAGIC             ) tmp_in
# MAGIC           WHERE TOT_GOALS IS NOT NULL AND TOT_GOALS <> 0
# MAGIC         ) tmp_out
# MAGIC       WHERE goals_rank <= 3 AND LOCATION_STR != 'Unknown' AND SHOT_PLACE_STR != 'Unknown'
# MAGIC       ORDER BY SHOT_PLACE_STR

# COMMAND ----------

# DBTITLE 1,Goals per time bin per country/league
# MAGIC %sql
# MAGIC   SELECT COUNTRY_CODE, TIME_BIN, COUNT(1) TOT_GOALS
# MAGIC    FROM GAME_EVENTS
# MAGIC    WHERE is_goal = 1
# MAGIC    GROUP BY COUNTRY_CODE, TIME_BIN
# MAGIC    ORDER BY COUNTRY_CODE, TIME_BIN

# COMMAND ----------

# MAGIC %md ### Third-party Visualization (Plotly)

# COMMAND ----------

# MAGIC %md 
# MAGIC Install the `plotly==2.0.15` pypi library and attach to the cluster. See the Databricks library documentation for instructions [Databricks](https://docs.databricks.com/user-guide/libraries.html#creating-libraries) | [Azure Databricks](https://docs.azuredatabricks.net/user-guide/libraries.html#create-a-library).

# COMMAND ----------

# DBTITLE 1,Verify that plotly library is accessible
# MAGIC %sh /databricks/python/bin/pip list --format=columns | grep plotly

# COMMAND ----------

# DBTITLE 1,Necessary imports
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go

import numpy as np
import json

# COMMAND ----------

# DBTITLE 1,Import helper transformation functions defined in another notebook
# MAGIC %run ./Helper_Functions

# COMMAND ----------

# DBTITLE 1,Prep data for viz - distribution of goals by shot placement and location
vizDf = spark.sql("SELECT shot_place_str, location_str, is_goal FROM GAME_EVENTS WHERE is_goal=1 AND location_str != 'NA' AND shot_place_str != 'NA'")
vizDf = vizDf.groupBy("shot_place_str","location_str").sum("is_goal").withColumnRenamed("sum(is_goal)","tot_goals").na.fill(0)
display(vizDf)

# COMMAND ----------

# DBTITLE 1,Convert DataFrame to JSON
vizlist = dfToJson(vizDf)
vizlist[:1]

# COMMAND ----------

# DBTITLE 1,Create the 3-D scatter plot
init_notebook_mode(connected=True)

x= jsonColList(vizlist, 'location_str', 0)
y= jsonColList(vizlist, 'tot_goals', 0)
z= jsonColList(vizlist, 'shot_place_str', 0)

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        color=z,
        colorscale='Viridis'
    ),
)

data = [trace1]
layout = go.Layout(
    autosize=True,
    width=1100,
    height=600,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
    xaxis = dict(title="X:Location"),
    yaxis = dict(title="Y:Goals"),
    zaxis = dict(title="Z:ShotPlace")
    )
)
fig = go.Figure(data=data, layout=layout)
displayHTML(plot(fig, filename='3d-scatter-colorscale', output_type='div'))