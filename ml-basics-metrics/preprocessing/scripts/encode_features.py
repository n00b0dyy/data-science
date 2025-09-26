import pandas as pd
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=columns)

def label_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label encode categorical columns (assign integers to categories)."""
    df_copy = df.copy()
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
    return df_copy
