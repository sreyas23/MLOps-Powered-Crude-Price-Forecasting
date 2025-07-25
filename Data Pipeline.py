# Databricks notebook source
#!pip install fredapi -q
from fredapi import Fred
import pandas as pd
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_fred_data(series_id, frequency='B'):
    """Fetch data from FRED and handle potential errors."""
    try:
        data = fred.get_series(series_id)
        df = pd.DataFrame(data, columns=[series_id])
        df = df.asfreq(frequency).ffill()
        df.index.name = 'Date'
        logging.info(f"Fetched {series_id} data successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching {series_id} data: {e}")
        return pd.DataFrame()

fred = Fred(api_key='a806a34e0f25d8e683d3a92913df5258')

brent_crude_df = fetch_fred_data('DCOILBRENTEU')
dxy_df = fetch_fred_data('DTWEXBGS')
gdp_df = fetch_fred_data('GDP')
cpi_df = fetch_fred_data('CPIAUCSL')
ipi_df = fetch_fred_data('INDPRO')             # Industrial Production Index
interest_rates_df = fetch_fred_data('FEDFUNDS')  # Federal Funds Rate
crude_oil_imports_df = fetch_fred_data('IR14280')  # U.S. Crude Oil Imports
crude_oil_exports_df = fetch_fred_data('IR14260')  # U.S. Crude Oil Exports
total_sales_df = fetch_fred_data('TOTALSA')  # Total Auto Sales


dataframes = [
    brent_crude_df, dxy_df, gdp_df, cpi_df, ipi_df, interest_rates_df,
    crude_oil_imports_df, crude_oil_exports_df, total_sales_df
]

if any(df.empty for df in dataframes):
    logging.error("One or more data series could not be fetched. Exiting script.")
    exit()


crude_oil_data = brent_crude_df.join([
    dxy_df, gdp_df, cpi_df, ipi_df, interest_rates_df,
    crude_oil_imports_df, crude_oil_exports_df, total_sales_df
], how='inner')

# COMMAND ----------

train_df = crude_oil_data.sample(frac=0.9, random_state=42)
test_df = crude_oil_data.drop(train_df.index)

# Swap the two features in test_df
test_df_changed = test_df.copy()
test_df_changed['DTWEXBGS'], test_df_changed['TOTALSA']  = test_df_changed['TOTALSA'], test_df['DTWEXBGS'] 

# Save the training and test data to CSV files
train_df.to_csv('training_data.csv', index=False)
test_df.to_csv('validation_data.csv', index=False)
test_df_changed.to_csv('validation_data_changed.csv', index=False)

# COMMAND ----------

table_df = spark.sql("SELECT * FROM train_data_table")
display(table_df)

view_df = spark.sql("SELECT * FROM validation_data_table")
display(view_df)

# COMMAND ----------


