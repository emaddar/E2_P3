import numpy as np
import pandas as pd
import missingno as msno

from scipy.stats import kurtosis, skew
import seaborn as sns

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




def calculate_category_occurrence(df):
    # Initialize an empty dictionary to store results
    result_dict = {}
    
    # Loop through each categorical column
    for col in df.columns:
        value_counts = df[col].value_counts()
        total_samples = df.shape[0]
        
        # Calculate the percentage occurrence of each category
        percentage_occurrence = (value_counts / total_samples) * 100
        
        # Store the percentage occurrence in the result dictionary
        result_dict[col] = percentage_occurrence
    
    # Convert the result dictionary to a DataFrame
    result_df = pd.DataFrame(result_dict)
    
    return result_df


def calculate_category_variables(df, threshold_min=95, threshold_max=100):
    category_variables_list = []

    for col in df.columns:
        if df[col].dtype == 'object':  # Assuming only categorical variables are of type 'object'
            total_count = df[col].count()
            category_counts = df[col].value_counts()
            category_percentages = (category_counts / total_count) * 100

            # Check if any category has a percentage within the specified range
            if any((category_percentages >= threshold_min) & (category_percentages <= threshold_max)):
                category_variables_list.append(col)

    return category_variables_list