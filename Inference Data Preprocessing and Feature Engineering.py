# Databricks notebook source
import pandas as pd

# Retrieve the parameter
condition = dbutils.widgets.get("condition")

print(f"Condition: {condition}")

# Define the DBFS input path based on the condition
dbfs_input_path = None
if condition == "original":
    dbfs_input_path = "/Workspace/Shared/validation_data.csv"
elif condition == "changed":
    dbfs_input_path = "/Workspace/Shared/validation_data_changed.csv"

# Ensure dbfs_input_path is defined
if dbfs_input_path is None:
    raise ValueError(f"Unexpected condition value: {condition}")

# Define the consistent output path
dbfs_output_path = "/Workspace/Shared/validation_data_processed.csv"

# Load the test data
df = pd.read_csv(dbfs_input_path)


# COMMAND ----------

#DCOILBRENTEU	DTWEXBGS	INDPRO	FEDFUNDS	TOTALSA	DXY_Change_1W	DXY_Change_1M	DXY_Change_1Q	Real_GDP	Interest_Rate_Diff
df['DollarIndex_1W_Chg'] = df['DTWEXBGS'].pct_change(periods=5) * 100
df['DollarIndex_1M_Chg'] = df['DTWEXBGS'].pct_change(periods=21) * 100
df['DollarIndex_1Q_Chg'] = df['DTWEXBGS'].pct_change(periods=63) * 100
df['Real_GDP'] = df['GDP'] / df['CPIAUCSL']  # This remains the same as it is a common title
df['IR_Diff'] = df['FEDFUNDS'] - df['IR14280']

df.rename(columns={
    'DCOILBRENTEU': 'Brent_Oil',
    'DTWEXBGS': 'DollarIndex',
    'GDP': 'GDP',
    'CPIAUCSL': 'CPI',
    'INDPRO': 'Ind_Prod',
    'FEDFUNDS': 'Fed_Funds',
    'TOTALSA': 'Total_AutoSales',
}, inplace=True)


for column in df.columns:
    df[column] = df[column].fillna(method='ffill')
df = df.drop(columns = ['GDP', 'CPI', 'IR14260', 'IR14280'])

df.head(25)
print(df.isnull().sum())

df = df.dropna()
print(df.isnull().sum())

# Save the processed data with a consistent file name
df.to_csv(dbfs_output_path, index=False)
