import os
import pandas as pd
import numpy as np

from feature_asymmetry import extract_asymmetry  
from feature_border import border
from feature_color import safe_color_features_extraction
from feature_diameter import diameter
from feature_evolution import evolution_score_row
from feature_fitzpatrick import fitzpatrick
from feature_fractal_lacunarity import lacunarity_for_masks
from feature_GLCM_contrast import contrast




data_path = '2026-PDS-Tigers/data'
df = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
mask_dir = os.path.join(data_path, "masks")
N_TEST = 2000

necessary_df = df[
    df["img_id"].apply(lambda x: os.path.exists(f"{mask_dir}/{x.removesuffix('.png')}_mask.png"))
][["img_id", "diagnostic"]].copy()


features_df = necessary_df.sample(n=N_TEST, random_state=42).copy()

### Asymmetry extraction
features_df['asymmetry'] = features_df['img_id'].apply(extract_asymmetry)

### Border features
features_df[["compactness", "convexity"]] = features_df["img_id"].apply(lambda x: pd.Series(border(x)))

### Color features
feature_cols = [
    "Ls_value", "as_value", "bs_value", "mean_angle_h", "s_value", "v_value",
    "r_value", "g_value", "b_value", "Ls_var", "as_var", "bs_var",
    "h_var", "s_var", "v_var", "r_var", "g_var", "b_var",
    "hsv_var_mean", "rgb_var_mean", "hsv_var_mag", "rgb_var_mag",
    "circular_max_min_h"
]
features_df[feature_cols] = features_df["img_id"].apply(lambda x: pd.Series(safe_color_features_extraction(x)))

#### Diameter extraction
features_df['diameter'] = features_df['img_id'].apply(diameter)

#### Evolution extraction
#features_df['evolution'] = features_df['img_id'].apply(evolution)

#### Fitzpatrick extraction
features_df['fitz'] = features_df['img_id'].apply(fitzpatrick)

#### Fractal lacunarity
features_df['lacunarity'] = features_df['img_id'].apply(lacunarity_for_masks)

#### Contrast extraction
features_df['contrast'] = features_df['img_id'].apply(contrast)


cancer_labels = ["BCC", "SCC", "MEL"]
features_df['cancerous'] = features_df['diagnostic'].isin(cancer_labels).astype(int)
features_df = features_df.drop("diagnostic", axis=1)

features_df.to_csv(os.path.join(data_path, "features.csv"), index=False)

print('Done')