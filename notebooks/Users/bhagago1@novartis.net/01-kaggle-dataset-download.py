# Databricks notebook source
# MAGIC %md
# MAGIC # Connecting to Kaggle via API and copying competition files to Azure Blob Storage
# MAGIC 
# MAGIC The Kaggle API allows us to connect to various competitions and datasets hosted on the platform: [API documentation](https://github.com/Kaggle/kaggle-api).<BR>
# MAGIC   
# MAGIC **Pre-requisite**: You should have downloaded the _kaggle.json_ containing the API *username* and *key* and localized the notebook below.<br><br>
# MAGIC In this notebook, we will -<br>
# MAGIC   1. Mount a container called `bronze` in Azure Blob Storage
# MAGIC   2. Import the competition data set in .zip format from Kaggle to the mounted container
# MAGIC   3. Unzip the downloaded data set and remove the zip file  

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Mount a container from Azure Blob Storage

# COMMAND ----------

# Replace with your storage account name
storageAccountName = "synapsestorageacct2"
storageAccountAccessKey = ""

# COMMAND ----------

# This is a function to mount a storage container
def mountStorageContainer(storageAccount, storageAccountKey, storageContainer, blobMountPoint):
  try:
    print("Mounting {0} to {1}:".format(storageContainer, blobMountPoint))
    # Unmount the storage container if already mounted
    dbutils.fs.unmount(blobMountPoint)
  except Exception as e:
    # If this errors, safe to assume that the container is not mounted
    print("....Container is not mounted; Attempting mounting now..")
    
  # Mount the storage container
  mountStatus = dbutils.fs.mount(
                  source = "wasbs://{0}@{1}.blob.core.windows.net/bronze/".format(storageContainer, storageAccount),
                  mount_point = blobMountPoint,
                  extra_configs = {"fs.azure.account.key.{0}.blob.core.windows.net".format(storageAccount): storageAccountKey})

  print("....Status of mount is: " + str(mountStatus))
  print() # Provide a blank line between mounts

# COMMAND ----------

# Mount the various storage containers created
mountStorageContainer(storageAccountName,storageAccountAccessKey,"synapsetestfs001","/mnt/bronze")

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/mnt/bronze

# COMMAND ----------

# Clean up from previous runs
dbutils.fs.rm("/mnt/bronze/us-census-demographic-data", recurse=True)
dbutils.fs.mkdirs("/mnt/bronze/us-census-demographic-data")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Import the competition data sets from Kaggle
# MAGIC 
# MAGIC We will be passing in the _KAGGLE\_USERNAME_ and _KAGGLE\_KEY_ parameters to os for the kaggle api to authenticate.

# COMMAND ----------

# Store Kaggle Credentials
import os
os.environ['KAGGLE_USERNAME'] = "gopinathbhagavathula"
os.environ['KAGGLE_KEY'] = ""

# COMMAND ----------

# Authenticate to Kaggle
import kaggle
kaggle.api.authenticate()

# COMMAND ----------

# Download the competition dataset to mounted container
!kaggle datasets download muonneutrino/us-census-demographic-data -p '/dbfs/mnt/bronze/us-census-demographic-data' -o

# COMMAND ----------

# Display downloaded file
display(dbutils.fs.ls("/mnt/bronze/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Unzip dataset and remove zip
# MAGIC 
# MAGIC We use the unix commands demonstrated in this notebook to unzip the _us-census-demographic-data.zip_ file and remove it: [Unzip Tutorial](https://docs.databricks.com/_static/notebooks/zip-files-python.html).<BR>

# COMMAND ----------

# MAGIC %sh
# MAGIC cd '/dbfs/mnt/bronze/us-census-demographic-data'
# MAGIC unzip '/dbfs/mnt/bronze/us-census-demographic-data/microsoft-malware-prediction.zip'

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r '/dbfs/mnt/bronze/'

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Return pass to Data Factory
# MAGIC 
# MAGIC We exit from this notebook and return "Pass" for Azure Data Factory.

# COMMAND ----------

dbutils.notebook.exit("Pass")
