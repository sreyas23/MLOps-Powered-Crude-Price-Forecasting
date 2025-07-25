# Databricks notebook source
import pandas as pd

dbfs_path = "training_data.csv"
df = pd.read_csv(dbfs_path)
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

# COMMAND ----------

print(df.isnull().sum())

# COMMAND ----------

df = df.dropna()
print(df.isnull().sum())

# COMMAND ----------

dbfs_path = dbfs_path.replace(".csv", '')
df.to_csv(f"{dbfs_path}_processed.csv", index=False)
