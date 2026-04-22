import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Calculate the correlation matrix
# We exclude the 'cancerous' label if it's a boolean/string, 
# or include it to see how features correlate with the target.
features_path = "2026-PDS-Tigers/data/features.csv"
features_df = pd.read_csv(features_path)

# Drop columns if they exist
features_df.drop(['fitz', 'img_id'], axis=1, inplace=True, errors='ignore')
features_df.dropna(inplace=True)


corr_matrix = features_df.corr()

# 2. Set up the matplotlib figure
plt.figure(figsize=(16, 12))

# 3. Create a mask to hide the upper triangle (optional but cleaner)
# Since the matrix is symmetrical, we only need to see the bottom half
import numpy as np
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 4. Generate the heatmap
sns.heatmap(
    corr_matrix, 
    mask=mask, 
    annot=True,          # Show the numbers in the cells
    fmt=".2f",           # Format to 2 decimal places
    cmap='coolwarm',     # Red for positive, Blue for negative correlation
    center=0,            # Ensure 0 is the neutral color
    linewidths=0.5, 
    cbar_kws={"shrink": .8}
)

plt.title("Feature Correlation Matrix", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 5. Save and show
plt.savefig("2026-PDS-Tigers/results/figures/correlation_matrix.png")
plt.show()