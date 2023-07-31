import numpy as np
import pandas as pd
import missingno as msno

from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt

def extended_describe(column_data):
    # Calculate kurtosis and skewness
    kurt = kurtosis(column_data)
    skewness = skew(column_data)

    # Convert original describe output to DataFrame
    stats = column_data.describe().to_frame()

    # Add kurtosis and skewness as new rows
    stats.loc['kurtosis'] = kurt
    stats.loc['skewness'] = skewness

    return stats