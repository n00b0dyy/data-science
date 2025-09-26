import pandas as pd


def find_dominant_columns(df: pd.DataFrame, threshold: float = 0.90) -> list:
    
    """
    Identify categorical columns where a single value dominates.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Share of the most common value above which
                           the column is flagged (default=0.95).
    
    Returns:
        list: Columns flagged for potential removal.
    """
    to_drop = []

    for col in df.select_dtypes(include="object").columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        top_value = value_counts.iloc[0]

        if top_value > threshold:
            to_drop.append(col)

    return to_drop



