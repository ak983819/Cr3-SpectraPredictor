import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

# Input file
input_file = Path(__file__).parent / "trainingset_em.xlsx"  # Keep this script and training dataset in the same folder

EV_NM_CONSTANT = 1239.8419843320026

# Load data
df = pd.read_excel(input_file)
df.columns = df.columns.str.strip()

target_nm_col = df.columns[1]
feature_cols = list(df.columns[3:])

df[target_nm_col] = pd.to_numeric(df[target_nm_col], errors="coerce")

if df[target_nm_col].isna().any():
    raise ValueError("Target column contains NaN values.")

if (df[target_nm_col] <= 0).any():
    raise ValueError("Target column contains zero or negative values.")

target_col = "Emission_eV_from_nm"
df[target_col] = EV_NM_CONSTANT / df[target_nm_col]

X = df[feature_cols].copy()
y = df[target_col].copy()

X = X.apply(pd.to_numeric, errors="coerce")

print("Shape of X:", X.shape)
print("Number of features:", X.shape[1])
print("Feature columns:", feature_cols)

# CatBoost parameters
params = {
    "iterations": 2092,
    "learning_rate": 0.024495702352497536,
    "depth": 4,
    "l2_leaf_reg": 2,
    "random_strength": 2.5013784369774434,
    "subsample": 0.8923775826709153,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=5)

pred_eV = np.zeros(len(df))

for train_idx, val_idx in kf.split(X):
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = CatBoostRegressor(**params)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=(X_val_scaled, y_val),
        use_best_model=True,
        early_stopping_rounds=100,
    )

    pred_eV[val_idx] = model.predict(X_val_scaled)

actual_nm = df[target_nm_col].values
pred_nm = EV_NM_CONSTANT / pred_eV

r2 = r2_score(actual_nm, pred_nm)
mae = mean_absolute_error(actual_nm, pred_nm)
rmse = np.sqrt(mean_squared_error(actual_nm, pred_nm))

print("\n10-fold CV performance")
print(f"R2   = {r2:.4f}")
print(f"MAE  = {mae:.4f} nm")
print(f"RMSE = {rmse:.4f} nm")

# Parity plot
plt.figure(figsize=(6, 6), dpi=300)

plt.scatter(
    actual_nm,
    pred_nm,
    s=28,
    c="#0F52BA",
    edgecolors="black",
    linewidths=0.5,
    alpha=0.85,
)

min_val = min(actual_nm.min(), pred_nm.min())
max_val = max(actual_nm.max(), pred_nm.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "--",
    color="#E37425",
    linewidth=1.2,
)

plt.xlabel("Actual Emission (nm)", fontsize=11)
plt.ylabel("Predicted Emission (nm)", fontsize=11)
plt.title("CatBoost 10-Fold CV Parity Plot", fontsize=12)

plt.text(
    0.05,
    0.95,
    f"R² = {r2:.3f}\nMAE = {mae:.3f} nm\nRMSE = {rmse:.3f} nm",
    transform=plt.gca().transAxes,
    va="top",
    fontsize=10,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)

plt.tick_params(
    axis="both",
    which="both",
    direction="in",
    length=4,
    width=0.8,
    top=True,
    right=True,
)

for spine in plt.gca().spines.values():
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.show()

# Predict new compounds
predict_file = Path(__file__).parent / "To_predict_em.xlsx"  # Keep this prediction file in the same folder as the script
output_file = Path(__file__).parent / "predicted_em.xlsx"    # Output will be saved in the same folder

predict_df = pd.read_excel(predict_file)
predict_df.columns = predict_df.columns.str.strip()

X_new = predict_df.iloc[:, 2:].copy()
X_new = X_new.apply(pd.to_numeric, errors="coerce")

if X_new.shape[1] != X.shape[1]:
    raise ValueError(
        f"Feature mismatch: training has {X.shape[1]} features, "
        f"prediction file has {X_new.shape[1]} features."
    )

X_new.columns = feature_cols

# Train final model on all training data
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_new_scaled = final_scaler.transform(X_new)

final_model = CatBoostRegressor(**params)
final_model.fit(X_scaled, y)

pred_new_eV = final_model.predict(X_new_scaled)
pred_new_nm = EV_NM_CONSTANT / pred_new_eV

result_df = predict_df.copy()
result_df["prediction"] = pred_new_nm

save_cols = ["Formula", "prediction", "SGR No.", "1/R2"]

missing_cols = [c for c in save_cols if c not in result_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in prediction output: {missing_cols}")

result_df[save_cols].to_excel(output_file, index=False)

print("\nTop 10 predicted emission results:")
print(result_df[save_cols].head(10).to_string(index=False))

print(f"\nPredictions saved to: {output_file}")
