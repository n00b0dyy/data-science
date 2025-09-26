import numpy as np
import pandas as pd

def normalize_missing(df: pd.DataFrame, column: str, missing_values=None) -> pd.DataFrame:
    """
    Normalize placeholders for missing values into proper NaN.
    
    Args:
        df (pd.DataFrame): input DataFrame
        column (str): column to normalize
        missing_values (list): list of values to be considered as missing
                               (default: ["", "?", "N/A", "NA", "nan", "none", "missing", "unknown", "Unknown"])
    
    Returns:
        pd.DataFrame: DataFrame with normalized missing values
    """
    if missing_values is None:
        missing_values = ["", "?", "N/A", "NA", "nan", "none", "missing", "unknown", "Unknown"]

    # Replace given placeholders with np.nan
    df[column] = df[column].replace(missing_values, np.nan)

    return df
