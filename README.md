#  MLOps powered Crude Oil Price Prediction on Databricks

## Overview  
This project focuses on predicting Brent Crude Oil prices using macroeconomic indicators and global events. The primary goal is to build an end-to-end machine learning pipeline in Databricks, leveraging MLOps principles for data preprocessing, model training, deployment, and monitoring.

Brent Crude Oil is a significant global economic indicator, affecting energy costs, transportation, and manufacturing. Our model aims to enhance forecasting accuracy and provide insights into the factors influencing price fluctuations.

## Objectives  
- Predict Brent Crude Oil prices using historical and macroeconomic data  
- Build scalable and reproducible MLOps pipelines in Databricks  
- Deploy models for real-time inference and monitoring  
- Evaluate the impact of feature changes (covariate shifts) on model performance  

## Key Features  

### Pipeline Implementation  
- Data preprocessing, including feature engineering  
- Automated model selection using H2O AutoML  
- Model deployment with MLflow  
- Model monitoring with Evidently AI  

### Feature Importance  
- Macro-economic variables like GDP, CPI, and Dollar Index  

### Experimentation  
- Tested impact of feature swaps (e.g., Dollar Index and Auto Sales) to simulate real-world changes  

## Methodology  

### 1. Data Collection and Preprocessing  
**Source**: Historical macroeconomic and Brent Crude Oil datasets  

### Exploratory Data Analysis (EDA)  
Analyzed variables like GDP, CPI, Federal Funds Rate, and Vehicle Sales  

#### Insights  
- Weak correlation between most macroeconomic factors and Brent Crude Oil prices  
- Strong negative correlation with the Dollar Index  

### 2. Model Training  
**AutoML Settings**  
- Runtime: 30 minutes  
- Evaluation Metric: RMSE  

**Selected Model**  
- Random Forest with hyperparameter optimization using Optuna  

**Performance Metrics**  
- RMSE (Cross-validation): 2.8802  

### 3. Deployment and Monitoring  
**Model Deployment**  
- Used MLflow for model tracking and real-time inference  

**Monitoring**  
- Pre- and post-feature swap monitoring using Evidently AI  
- Metrics include Covariate Shift and Prior Probability Shift  

### 4. Feature Engineering  
Custom features developed:  
- Dollar Change Calculations (1-week, 1-month, and 1-quarter trends)  
- Real GDP adjustments using CPI  
- Interest Rate Differentials for spread/yield curve modeling  

## Results  

**Model Performance**  
- Best Model: Random Forest  
- RMSE: 2.8802 (after optimization)  

**Feature Impact**  
- Dollar Index: Most influential feature with strong negative correlation  
- Car Sales: Weak correlation with oil prices  

**Monitoring**  
- Covariate shift detected after feature swaps, affecting prediction accuracy  

## Files  

**AutoML.py**  
Script for running automated machine learning using H2O AutoML to identify the best-performing models for Brent Crude Oil price prediction  

**Data Pipeline.py**  
Contains the preprocessing pipeline for raw data, including cleaning, feature selection, and dataset preparation  

**Data Preprocessing and Feature Engineering.py**  
Includes detailed steps for feature extraction and transformation, such as calculating dollar change trends and real GDP adjustments  

**Inference Data Preprocessing and Feature Engineering.py**  
Handles preprocessing for the test data during model inference, including feature transformations and adjustments  

**Model Inference and Monitoring.py**  
Implements model inference and monitoring tasks, including evaluating predictions and detecting covariate shifts  

**Model Training and Deployment.py**  
Combines model training and deployment using MLflow, ensuring reproducibility and scalability of the pipeline  

**actual_vs_predicted.png**  
Visual comparison of actual vs. predicted Brent Crude Oil prices before and after feature swaps, demonstrating the model’s performance  

**actual_vs_predicted (After Swapping Features).png**  
Shows how feature swaps impacted the model’s predictions, including visualization of covariate shifts  

**training_data_processed.csv**  
The preprocessed training dataset used for model development and optimization  

**validation_data.csv**  
Validation dataset used to test model performance and fine-tune hyperparameters  

**validation_data_changed.csv**  
Validation dataset with altered feature values (e.g., feature swaps) to simulate covariate shifts and evaluate robustness  

**validation_data_processed.csv**  
Fully preprocessed validation dataset used for monitoring and inference  
