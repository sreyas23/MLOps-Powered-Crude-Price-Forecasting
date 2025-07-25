# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import optuna
import pickle
import os
import mlflow
import mlflow.sklearn


# COMMAND ----------

df = pd.read_csv("/Workspace/Shared/training_data_processed.csv")

print("Data loaded. Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# COMMAND ----------

def train_and_optimize_model(df, target_column, n_trials=50, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_uniform('max_features', 0.1, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        rmse = mean_squared_error(y_test, preds, squared=False)
        
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print("Best Hyperparameters: ", best_params)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_scaled, y)
    
    return best_model, best_params, scaler

# COMMAND ----------

best_model, best_params, scaler = train_and_optimize_model(df, 'Brent_Oil')

# COMMAND ----------

# MAGIC %md
# MAGIC Best model RMSE: 2.880150000148329
# MAGIC

# COMMAND ----------

best_model

# COMMAND ----------

best_params

# COMMAND ----------

with mlflow.start_run() as run:
    # Log the model
    mlflow.sklearn.log_model(best_model, "best_model")
    
    # Log the hyperparameters
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    
    # Register the model
    model_uri = "runs:/{}/best_model".format(run.info.run_id)
    mlflow.register_model(model_uri, "BestModel")
