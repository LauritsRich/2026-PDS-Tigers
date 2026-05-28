import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


prediction_results_path = "2026-PDS-Tigers/results/predictions/predictions"

knn_data = pd.read_csv(f"{prediction_results_path}_knn_extended.csv")
rf_data = pd.read_csv(f"{prediction_results_path}_rf_extended.csv")
lg_data = pd.read_csv(f"{prediction_results_path}_lg_extended.csv")

knn_pred = knn_data[['img_id','probability','prediction']].copy()
knn_pred = knn_pred.rename(columns={'probability': 'knn_probability', 'prediction':'knn_prediction'})
rf_pred = rf_data[['img_id', 'probability', 'prediction']].copy()
rf_pred = rf_pred.rename(columns={'probability': 'rf_probability', 'prediction':'rf_prediction'})
lg_pred = lg_data[['img_id', 'probability', 'prediction', 'actual']].copy()
lg_pred = lg_pred.rename(columns={'probability': 'lg_probability', 'prediction':'lg_prediction'})


result = knn_pred.merge(rf_pred, on='img_id').merge(lg_pred, on='img_id')
#### VOTING SYSTEM, METHOD 1: EXPONENTIAL APPROACH

best_auc = 0
best_p = None
powers = np.linspace(1, 7, 28)
aucs = []

for p in powers:  # Try different power levels for "punishing" the uncertainty
    score_1 = (result['knn_probability'] ** p +
               result['rf_probability'] ** p +
               result['lg_probability'] ** p)

    score_0 = ((1 - result['knn_probability']) ** p +
               (1 - result['rf_probability']) ** p +
               (1 - result['lg_probability']) ** p)

    p_final = score_1 / (score_1 + score_0)

    auc = roc_auc_score(result['actual'], p_final)
    aucs.append(auc)

    if auc > best_auc:
        best_auc = auc
        best_p = p


# Plot
# plt.figure(figsize=(7,5))
# plt.plot(powers, aucs, marker='o')
# plt.axvline(best_p, linestyle='--', label=f'Best p = {best_p:.2f}')
# plt.axhline(best_auc, linestyle='--', label=f'Best AUC = {best_auc:.4f}')

# plt.xlabel('Power (p)')
# plt.ylabel('AUC')
# plt.title('AUC vs Power in Voting Ensemble')
# plt.legend()
# plt.grid(True)
# plt.show()

print(round(best_p, 4))



result['method_1_prob'] = p_final


auc_knn = roc_auc_score(result['actual'], result['knn_probability'])
auc_rf = roc_auc_score(result['actual'], result['rf_probability'])
auc_lg = roc_auc_score(result['actual'], result['lg_probability'])
auc_1 = roc_auc_score(result['actual'], result['method_1_prob'])

print(f"KNN AUC:{auc_knn:.4f}")
print(f"RF AUC:{auc_rf:.4f}")
print(f"LG AUC:{auc_knn:.4f}")
print(f"METHOD 1 AUC: {auc_1:.4f}")


#### VOTING SYSTEM, METHOD 3: MOST CONFIDENT
result['most_confident'] = result[['knn_probability', 'rf_probability', 'lg_probability']].apply(
    lambda row: row.loc[(row - 0.5).abs().idxmax()],
    axis=1
)

auc_3 = roc_auc_score(result['actual'], result['most_confident'])
print(f"METHOD 3 AUC: {auc_3:.4f}")

print(result.head())



