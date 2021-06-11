# Databricks notebook source
!pip install jina

# COMMAND ----------

import numpy as np
from jina import Document

d0 = Document(id='🐲', embedding=np.array([0, 0]))
d1 = Document(id='🐦', embedding=np.array([1, 0]))
d2 = Document(id='🐢', embedding=np.array([0, 1]))
d3 = Document(id='🐯', embedding=np.array([1, 1]))



# COMMAND ----------

d0.chunks.append(d1)
d0.chunks[0].chunks.append(d2)
d0.matches.append(d3)



# COMMAND ----------

ht = (d0.plot())

# COMMAND ----------

ht

# COMMAND ----------

displayHTML(d0.plot())