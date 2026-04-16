import pandas as pd
import numpy as np
def to_binary_feature(value):
    
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ["true", "yes", "1", "sim"]:
            return 1
        if value in ["false", "no", "0", "não", "nao"]:
            return 0
    return int(bool(value))

def extract_features_fast(row):
    img_id = row["img_id"]

    
    mask = load_mask(img_id, data_path)
    asymmetry = get_asymmetry(mask)

    feats = {
        "asymmetry": asymmetry,
        "grew": to_binary_feature(row.get("grew", np.nan)),
        "changed": to_binary_feature(row.get("changed", np.nan)),
        "diagnostic": row["diagnostic"],
    }
    return feats

def evolution_score_row(row):
    try:
        vals=[]
        for col in ["grew", "changed"]:
            v = to_binary_feature(row.get(col, np.nan))
            if not pd.isna(v):
                vals.append(v)
        if len(vals) == 0:
            return np.nan
        return float(np.sum(vals))
    except:
        return np.nan