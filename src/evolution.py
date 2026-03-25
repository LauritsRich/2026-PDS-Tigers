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
    cols = ["grew", "changed"]
    vals = [row[c] for c in cols if pd.notna(row[c])]
    if len(vals) == 0:
        return np.nan
    return np.sum(vals)