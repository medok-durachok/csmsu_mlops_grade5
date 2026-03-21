import numpy as np
import pandas as pd

def basic_dq_report(df):
    n = len(df)
    dup_rate = float(df.duplicated().mean()) if n else 0.0
    missing_rate = df.isna().mean().to_dict()
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    types = {c: str(df[c].dtype) for c in df.columns}

    return {
        "n_rows": int(n),
        "duplicate_rate": dup_rate,
        "missing_rate": {k: float(v) for k, v in missing_rate.items()},
        "constant_cols": const_cols,
        "dtypes": types,
    }
