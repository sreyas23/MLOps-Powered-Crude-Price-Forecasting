# Databricks notebook source
! apt-get install default-jre
!java -version

# COMMAND ----------

pip install --upgrade pip --quiet


# COMMAND ----------

!pip install h2o --quiet

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import h2o
from h2o.automl import H2OAutoML
import mlflow
import mlflow.sklearn
import os

# COMMAND ----------

df = pd.read_csv("/Workspace/Shared/training_data_processed.csv")

print("Data loaded. Shape:", df.shape)


# COMMAND ----------

df

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = df.drop('Brent_Oil', axis=1)
y = df['Brent_Oil']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# COMMAND ----------

h2o.init()
h2o_train = h2o.H2OFrame(pd.DataFrame(X_train_scaled, columns=X.columns).assign(Brent_Oil=y_train.values))
h2o_test = h2o.H2OFrame(pd.DataFrame(X_test_scaled, columns=X.columns).assign(Brent_Oil=y_test.values))

x = h2o_train.columns[:-1]
y = 'Brent_Oil'

h2o_automl = H2OAutoML(max_runtime_secs = 1800, sort_metric="rmse", exclude_algos =['StackedEnsemble'])
h2o_automl.train(x=x, y=y, training_frame=h2o_train)

leaderboard = h2o.automl.get_leaderboard(h2o_automl, extra_columns = "ALL")
print(leaderboard)

# COMMAND ----------

best_model = h2o_automl.leader
performance = best_model.model_performance()
r2 = performance.r2()
mae = performance.mae()
rmse = performance.rmse()

# COMMAND ----------

best_model

# COMMAND ----------

def save_feature_importance_plot(best_model, output_dir='plots'):
    if hasattr(best_model, 'varimp'):
        varimp = best_model.varimp(use_pandas=True)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='percentage', y='variable', data=varimp)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nFeature importance plot saved as {output_path}")


# COMMAND ----------

save_feature_importance_plot(best_model)


# COMMAND ----------


