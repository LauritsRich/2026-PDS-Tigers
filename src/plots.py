# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load and clean data
# features_path = "2026-PDS-Tigers/data/features.csv"
# features_df = pd.read_csv(features_path)

# # Drop column if it exists, ignoring errors if it's already gone
# features_df.drop('fitz', axis=1, inplace=True, errors='ignore')
# features_df.drop('img_id', axis=1, inplace=True, errors='ignore')
# features_df.dropna(inplace=True)


# cancerous = features_df[features_df['cancerous'] == True]
# not_cancerous = features_df[features_df['cancerous'] == False]

# features_list = features_df.columns.to_list()
# features_list.remove('cancerous')

# for i in range(len(features_list)):
#     sns.kdeplot(cancerous[features_list[i]], label="Cancerous")
#     sns.kdeplot(not_cancerous[features_list[i]], label="Non-Cancerous")
#     plt.legend()
#     plt.title(features_list[i])
#     plt.savefig(f"2026-PDS-Tigers/results/figures/{features_list[i]}_kde_plot.png")
#     plt.close()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint


# Function to calculate univariate Mahalanobis distance between two groups
def calculate_mahalanobis(group1, group2):
    mu1, mu2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled variance
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    
    # If variance is 0, distance is 0 if means are equal, else undefined (return 0 or handle)
    if pooled_var == 0:
        return 0.0
    
    # Mahalanobis Distance (standardized difference)
    distance = np.abs(mu1 - mu2) / np.sqrt(pooled_var)
    return distance

# Load and clean data
features_path = "2026-PDS-Tigers/data/features.csv"
features_df = pd.read_csv(features_path)

# Drop columns if they exist
features_df.drop(['fitz', 'img_id'], axis=1, inplace=True, errors='ignore')
features_df.dropna(inplace=True)

# Split into classes
cancerous = features_df[features_df['cancerous'] == True]
not_cancerous = features_df[features_df['cancerous'] == False]

# Get feature names
features_list = [col for col in features_df.columns if col != 'cancerous']

# Dictionary to store results
mahalanobis_dict = {}



for feature in features_list:
    # Extract data for the current feature
    c_data = cancerous[feature]
    nc_data = not_cancerous[feature]
    
    # Compute Mahalanobis Distance
    dist = calculate_mahalanobis(c_data, nc_data)
    mahalanobis_dict[feature] = dist
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.kdeplot(c_data, label="Cancerous", fill=True)
    sns.kdeplot(nc_data, label="Non-Cancerous", fill=True)
    
    plt.legend()
    # Add distance to title
    plt.title(f"Feature: {feature}\nMahalanobis Distance: {dist:.4f}")
    plt.xlabel(feature)
    plt.ylabel("Density")
    
    # Save the plot
    plt.savefig(f"2026-PDS-Tigers/results/figures/{feature}_kde_plot.png")
    plt.close()

# Return/Print the dictionary
print("Mahalanobis Distances for each feature:")


threshold = 0.2

selected_features = [feat for feat, dist in mahalanobis_dict.items() if dist >= threshold]

print(f"Original feature count: {len(mahalanobis_dict)}")
print(f"Selected feature count: {len(selected_features)}")
print(f"Features kept: {selected_features}")

# Create your new filtered dataframe
filtered_df = features_df[selected_features + ['cancerous']]
pprint.pprint(filtered_df)







