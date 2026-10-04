import pandas as pd
import numpy as np

MONTH_START = 7

def load_data(path):
    df = pd.read_excel(path)
    month_cols = list(df.columns[MONTH_START:])
    for c in month_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, month_cols

def build_features(df, month_cols):
    out = df.copy()
    first7, last3 = month_cols[:7], month_cols[7:]
    out["train_avg"] = out[first7].mean(axis=1)
    out["train_std"] = out[first7].std(axis=1)
    out["train_min"] = out[first7].min(axis=1)
    out["train_max"] = out[first7].max(axis=1)
    out["train_total"] = out[first7].sum(axis=1, min_count=1)
    out["future_avg"] = out[last3].mean(axis=1)
    out["future_total"] = out[last3].sum(axis=1, min_count=1)
    out["cv"] = out["train_std"] / out["train_avg"].replace(0, np.nan)
    out["consumption_per_kw"] = out["train_avg"] / out["CONTRACT_LOAD"].replace(0, np.nan)
    return out, first7, last3

def make_target(df):
    threshold = df["train_avg"].quantile(0.75)
    out = df.copy()
    out["high_consumption_target"] = (out["future_avg"] >= threshold).astype(int)
    return out, threshold
