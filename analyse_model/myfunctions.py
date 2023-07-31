import numpy as np
import pandas as pd
import missingno as msno

from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt

def extended_describe_all_columns(df):
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Calculate kurtosis and skewness for the column
        kurt = kurtosis(df[column])
        skewness = skew(df[column])

        # Describe the column and add kurtosis and skewness as new rows
        column_stats = df[column].describe().to_frame()
        column_stats.loc['kurtosis'] = kurt
        column_stats.loc['skewness'] = skewness

        # Add the column's stats to the result DataFrame
        result_df[column] = column_stats[column]

    return result_df



def missing_values_summary(df):
    # Calculate the count of missing values per column
    missing_values_count = df.isna().sum()

    # Calculate the percentage of missing values per column
    total_rows = len(df)
    percentage_missing = (missing_values_count / total_rows) * 100

    # Create a DataFrame with missing values count and percentage
    result = pd.concat([missing_values_count, percentage_missing], axis=1, keys=['Missing Count', 'Percentage Missing %'])

    # Sort the values in descending order
    result_sorted = result.sort_values(by='Percentage Missing %', ascending=False)

    return result_sorted