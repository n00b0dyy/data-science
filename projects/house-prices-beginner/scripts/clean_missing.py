import pandas as pd
import numpy as np

def fill_missing(
    df: pd.DataFrame, 
    column: str, 
    strategy: str = "median", 
    fill_value=None, 
    groupby_col: str = None
) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame column with various strategies.

    Args:
        df (pd.DataFrame): input DataFrame
        column (str): column name to process
        strategy (str): filling method ("mean", "median", "mode", "constant",
                        "drop", "interpolate", "ffill", "bfill", 
                        "geomean", "groupby")
        fill_value (Any): value to use when strategy="constant"
        groupby_col (str): column used for group-based filling (when strategy="groupby")

    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """

    col_type = df[column].dtype

    # ===== Numeric strategies =====
    if strategy == "mean" and pd.api.types.is_numeric_dtype(col_type):
        # Replace NaN with column mean
        df[column] = df[column].fillna(df[column].mean())

    elif strategy == "median" and pd.api.types.is_numeric_dtype(col_type):
        # Replace NaN with column median
        df[column] = df[column].fillna(df[column].median())

    elif strategy == "geomean" and pd.api.types.is_numeric_dtype(col_type):
        # Replace NaN with geometric mean (only works for positive values).
        # ⚠️ Warning:
        # This strategy fills ONLY NaN values.
        # Non-positive values (0 or negative) are ignored and left unchanged,
        # because in many domains they may be valid (e.g., losses, temperatures).
        # If you want to treat non-positive values as missing, you must handle that explicitly.
        positive_vals = df[column][df[column] > 0]
        if len(positive_vals) == 0:
            raise ValueError("No positive values found for geometric mean")
        geo_mean = np.exp(np.log(positive_vals).mean())
        df[column] = df[column].fillna(geo_mean)

    # ===== Categorical or general strategies =====
    elif strategy == "mode":
        # Replace NaN with the most frequent value in the column.
        modes = df[column].mode()
        if modes.empty:
            raise ValueError(f"Cannot compute mode: column '{column}' has only NaN")
        df[column] = df[column].fillna(modes[0])

    elif strategy == "constant":
        # Replace NaN with a constant value provided by user
        if fill_value is None:
            raise ValueError("For strategy='constant' you must provide fill_value")
        df[column] = df[column].fillna(fill_value)

    elif strategy == "drop":
        # Drop rows where this column has NaN
        # ⚠️ Warning: this reduces the number of rows in the DataFrame.
        # In a pipeline, dropping rows here can cause index misalignment
        # with later operations if other columns are processed separately.
        df = df.dropna(subset=[column])

    elif strategy == "interpolate" and pd.api.types.is_numeric_dtype(col_type):
        # Replace NaN with interpolated values (linear by default)
        df[column] = df[column].interpolate()

    # ===== Datetime strategies =====
    elif strategy == "ffill":
        # Forward fill: copy last valid value forward
        df[column] = df[column].ffill()
        # If first value is still NaN, fill it backward
        if pd.isna(df[column].iloc[0]):
            df[column] = df[column].bfill()

    elif strategy == "bfill":
        # Backward fill: copy next valid value backward
        df[column] = df[column].bfill()
        # If last value is still NaN, fill it forward
        if pd.isna(df[column].iloc[-1]):
            df[column] = df[column].ffill()

    # ===== Group-based filling =====
    elif strategy == "groupby":
        # Fill NaN within groups defined by groupby_col.
        # ⚠️ Warning:
        # If the grouping column (groupby_col) contains NaN,
        # those rows will stay unfilled because NaN forms its own empty group.
        # In practice, make sure to clean or fill groupby_col first.
        if groupby_col is None:
            raise ValueError("For strategy='groupby' you must provide groupby_col")
        df[column] = df.groupby(groupby_col)[column].transform(
            lambda x: x.fillna(
                x.median() if pd.api.types.is_numeric_dtype(col_type) else x.mode()[0]
            )
        )

    else:
        raise ValueError(f"Strategy {strategy} not supported for type {col_type}")

    return df
