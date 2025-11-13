import numpy as np
import pandas as pd

def mask_pre_ipo(df):
    # For each stock, treat leading zeros as NA
    df2 = df.copy()
    for col in df2:
        s = df2[col]
        first_nonzero = s.ne(0).idxmax()  # first non-zero return
        df2.loc[:first_nonzero, col] = np.nan
    return df2

def safe_rolling_zscore(df, window):
    rolling_mean = df.rolling(window).mean()
    rolling_std = df.rolling(window).std()

    # If std == 0 → return 0 instead of NaN or inf
    z = (df - rolling_mean) / rolling_std.replace(0, np.nan)
    z = z.fillna(0)

    return z
