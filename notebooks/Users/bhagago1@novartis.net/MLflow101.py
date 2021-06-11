# Databricks notebook source
import numpy as np
import matplotlib.pyplot as plt 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# mlflow.sklearn provides API for logging and loading sklean models 
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the data

# COMMAND ----------

# load the dataset
data = spark.sql("select * from diabetes").toPandas()
data.head(3)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Set experiment in MLflow

# COMMAND ----------

# set MLflow experiment, if the experiment does not exist, create a new experiment
# experiment name must be unique and case sensitive
experiment_name = "/Predict-Diabetes"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment = experiment if experiment else mlflow.create_experiment(experiment_name)

print(experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set run in the experiment and build the model
# MAGIC Log paramters, metrics, and artifacts
# MAGIC Save the model as MLflow model

# COMMAND ----------

# prepare the dataset
features = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    "Age"
]

X, y = data[features].values, data["Diabetic"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# COMMAND ----------

# train a logistic regression model
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    # log parameters
    penalty = "l2"
    mlflow.log_param("Penalty", penalty)
    
    reg = 0.2
    mlflow.log_param("Regularization strength", reg)
    
    solver = "liblinear"
    mlflow.log_param("Solver", solver)
    
    model = LogisticRegression(C=1/reg, solver=solver, penalty=penalty).fit(X_train, y_train)

    # get accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    
    mlflow.log_metric("Accuracy", np.float(acc))
    
    # roc curve
    scores = model.predict_proba(X_test)[:,1]
    roc_auc = metrics.roc_auc_score(y_test, scores)

    mlflow.log_metric("AUC", roc_auc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.jpg")
    
    mlflow.log_artifact("roc_curve.jpg")

    # save the MLflow model
    model_path = "model"
    
    # infer model input/output signature
    signature = infer_signature(X_test, model.predict(X_test))
    mlflow.sklearn.log_model(model, model_path, signature=signature)
    
    print("Model URI")
    print(f"runs:/{run.info.run_id}/{model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Register the model on Databricks MLflow registry
# MAGIC Deploy the model as REST API endpoint as well.

# COMMAND ----------

# register model
# provide above Model URI and model name on registry
# if model name not available then it creates, otherwise creates new version the model

# mlflow.register_model(f"runs:/{run.info.run_id}/{model_path}", "predict-diabetes")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build model Docker image on AzureML

# COMMAND ----------

import mlflow.azureml
from azureml.core import Workspace

# COMMAND ----------

# MAGIC %%writefile ws_config.json
# MAGIC {
# MAGIC     "subscription_id": "d5d90bd0-0fd4-41ce-a604-cd1f847396d3_bhagago1",
# MAGIC     "resource_group": "DSAI_SBX_RESOURCEGROUP",
# MAGIC     "workspace_name": "dsai-mlw-001"
# MAGIC }

# COMMAND ----------

# create azureml workspace
ws = Workspace.from_config(path="./ws_config.json")

# COMMAND ----------

# build image
# resp = mlflow.azureml.build_image(f"runs:/{run.info.run_id}/{model_path}", ws, image_name='dialr', model_name="dialrmodel", synchronous=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Deploy the model on AzureML + Kubernetes

# COMMAND ----------

from azureml.core.webservice import aks

# COMMAND ----------

aks_config = aks.AksServiceDeploymentConfiguration(
    autoscale_enabled=True,
    autoscale_min_replicas=1,
    autoscale_max_replicas=5,
    autoscale_refresh_seconds=1,
    autoscale_target_utilization=70,
    collect_model_data=False,
    auth_enabled=False,
    cpu_cores=0.5,
    memory_gb=0.5,
    enable_app_insights=False,
    scoring_timeout_ms=60000,
    replica_max_concurrent_requests=1,
    max_request_wait_time=500,
    num_replicas=None,
    primary_key=None,
    secondary_key=None,
    tags=None,
    properties=None,
    description="",
    gpu_cores=None,
    period_seconds=1,
    initial_delay_seconds=310,
    timeout_seconds=1,
    success_threshold=1,
    failure_threshold=3,
    namespace="aks-inf-cluster",
    token_auth_enabled=False,
    compute_target_name="aks-inf-cluster",
    cpu_cores_limit=0.5,
    memory_gb_limit=0.5
)

# validate configuration
aks_config.validate_configuration()

# COMMAND ----------

# deploy the model on AzureML
mlflow.azureml.deploy("runs:/e2926c365f684e32983be309e84675f6/model", ws, deployment_config=aks_config, service_name="dialrpred", model_name="dialr", synchronous=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Access the model API deployed on Kubernetes

# COMMAND ----------

import json
import requests

# COMMAND ----------

query = [[0,171,80,34,23,43.50972593,1.213191354,21], [8,92,93,47,36,21.24057571,0.158364981,23], [9,103,78,25,304,29.58219193,1.282869847,43]]
json_query = json.dumps({"data": query})

print(json_query)

# request_headers = {
#     "Content-Type":"application/json",
#     "Authorization":"Bearer " + 'cfaenlolzSzxxxxxxxxxxxxX4ahil3Ulqz'
# }
request_headers = {
    "Content-Type":"application/json"
}

# COMMAND ----------

response = requests.post(
    url='http://51.145.182.161:80/api/v1/service/dialrpred/score',
    data=json_query,
    headers=request_headers)

response

# COMMAND ----------

response.json()

# COMMAND ----------

# MAGIC %%writefile test.py
# MAGIC import json
# MAGIC import requests
# MAGIC query = [[0,171,80,34,23,43.50972593,1.213191354,21], [8,92,93,47,36,21.24057571,0.158364981,23], [9,103,78,25,304,29.58219193,1.282869847,43]]
# MAGIC json_query = json.dumps({"data": query})
# MAGIC 
# MAGIC request_headers = {
# MAGIC     "Content-Type":"application/json"
# MAGIC }
# MAGIC 
# MAGIC for i in range(1000):
# MAGIC     response = requests.post(
# MAGIC         url='http://51.145.182.161:80/api/v1/service/dialrpred/score',
# MAGIC         data=json_query,
# MAGIC         headers=request_headers)
# MAGIC print(response.json())

# COMMAND ----------

# MAGIC %%writefile run.sh
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &
# MAGIC python test.py &

# COMMAND ----------

# MAGIC %sh
# MAGIC bash run.sh

# COMMAND ----------

