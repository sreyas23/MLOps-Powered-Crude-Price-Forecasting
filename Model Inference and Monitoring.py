# Databricks notebook source
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping
from evidently.ui.dashboards import DashboardPanelPlot, ReportFilter, PanelValue, TestFilter, TestSuitePanelType, DashboardPanelTestSuite, PlotType

# COMMAND ----------

train_data = pd.read_csv("/Workspace/Shared/training_data_processed.csv")
test_data = pd.read_csv("/Workspace/Shared/validation_data_processed.csv")
X_train_data = train_data.drop(["Brent_Oil"], axis=1)
X_test_data = test_data.drop(["Brent_Oil"], axis=1)
y_test = test_data["Brent_Oil"]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Inference

# COMMAND ----------

#Best Model Deployed
model_name = "BestModel"
model_uri = f"models:/{model_name}/1" 
model = mlflow.pyfunc.load_model(model_uri)

X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_data.columns)

# Perform inference
predictions = model.predict(X_test_scaled_df)

# Calculate RMSE
rmse = mean_squared_error(y_test, predictions, squared=False)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, predictions)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Values', color='b')
plt.plot(predictions, label='Predicted Values', color='r', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Brent Oil')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.savefig("actual_vs_predicted.png")

with mlflow.start_run():
    # Log the RMSE and MAPE
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    # Log the plot
    mlflow.log_artifact("actual_vs_predicted.png")


# COMMAND ----------

# MAGIC %md
# MAGIC # Model Monitoring 

# COMMAND ----------

#Set Evidently Cloud

token = 'dG9rbgEnlfEZum1MqKgic4fjIqQuBM5PusIqRUR/rioweY52BQBQ8Xg5woa72wmBaH41PAUVkFhtDgemf1awylEDVQ/1pwFNbBQG0FNHAc4Dp6bWH84DtdhiiZ1RdklOJOel3qD+rNrU41Jw3KKjNltsMHiiYuqlL+uW'

# Create CloudWorkspace
ws = CloudWorkspace(
    token=token,
    url="https://app.evidently.cloud"
)

project = ws.get_project("b219d140-f5b6-4880-9eee-0d656da5a1ff")

# Prepare data for Evidently AI
reference_data = pd.DataFrame(X_train_scaled, columns=X_train_data.columns)
reference_data['Brent_Oil'] = train_data['Brent_Oil'].values

current_data = pd.DataFrame(X_test_scaled, columns=X_test_data.columns)
current_data['Brent_Oil'] = predictions

#Column Mapping
column_mapping = ColumnMapping(
    prediction=None, 
    target="Brent_Oil"
)

drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

# Upload the report to Evidently AI Cloud
ws.add_report(
    report=drift_report,
    project_id=project.id
)

project.save()
