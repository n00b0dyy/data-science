import pandas as pd
from scripts.clean_missing import fill_missing
from scripts.normalize_missing import normalize_missing

# Complex DataFrame with various dirty values
df = pd.DataFrame({
    "Age": [22, None, 35, None, 40, 0, -5],                  # numeric, with None, 0, -5
    "Name": ["Alice", None, "Charlie", "?", "David", "N/A", ""],  # text with placeholders
    "Date": [pd.NaT, pd.Timestamp("2020-01-02"), None, "N/A", pd.Timestamp("2020-01-04"), "?", ""]  # mixed placeholders
})

print("=== ORIGINAL DATAFRAME ===")
print(df, "\n")

# 1. Normalize placeholders
df_norm = df.copy()
df_norm = normalize_missing(df_norm, "Name")
df_norm = normalize_missing(df_norm, "Date")

print("=== AFTER NORMALIZATION (placeholders -> NaN) ===")
print(df_norm, "\n")

# 2. Median
df_median = fill_missing(df_norm.copy(), "Age", strategy="median")
print("=== STRATEGY: median (Age) ===")
print(df_median, "\n")

# 3. Mean
df_mean = fill_missing(df_norm.copy(), "Age", strategy="mean")
print("=== STRATEGY: mean (Age) ===")
print(df_mean, "\n")

# 4. Mode
df_mode = fill_missing(df_norm.copy(), "Name", strategy="mode")
print("=== STRATEGY: mode (Name) ===")
print(df_mode, "\n")

# 5. Constant
df_const = fill_missing(df_norm.copy(), "Name", strategy="constant", fill_value="Unknown")
print("=== STRATEGY: constant (Name) ===")
print(df_const, "\n")

# 6. Drop
df_drop = fill_missing(df_norm.copy(), "Age", strategy="drop")
print("=== STRATEGY: drop (Age) ===")
print(df_drop, "\n")

# 7. Interpolate
df_interp = fill_missing(df_norm.copy(), "Age", strategy="interpolate")
print("=== STRATEGY: interpolate (Age) ===")
print(df_interp, "\n")

# 8. Forward fill (ffill)
df_ffill = fill_missing(df_norm.copy(), "Date", strategy="ffill")
print("=== STRATEGY: ffill (Date) ===")
print(df_ffill, "\n")

# 9. Backward fill (bfill)
df_bfill = fill_missing(df_norm.copy(), "Date", strategy="bfill")
print("=== STRATEGY: bfill (Date) ===")
print(df_bfill, "\n")

# 10. Geometric mean
df_geo = fill_missing(df_norm.copy(), "Age", strategy="geomean")
print("=== STRATEGY: geomean (Age) ===")
print(df_geo, "\n")

# 11. Groupby (fill Age within groups of Name)
# For demonstration, we artificially create missing values again
df_group = df_norm.copy()
df_group.loc[1, "Age"] = None
df_group.loc[3, "Age"] = None
df_group = fill_missing(df_group, "Age", strategy="groupby", groupby_col="Name")
print("=== STRATEGY: groupby (Age by Name) ===")
print(df_group, "\n")
