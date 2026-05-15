import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


input_file = Path.cwd() / "trainingset_fwhm.xlsx"  # keep this notebook and Excel file in the same folder

HC = 1239.841984

df = pd.read_excel(input_file)
df.columns = df.columns.astype(str).str.strip()

emission_nm_col = df.columns[1]
target_col = df.columns[2]
feature_cols = list(df.columns[3:])

df[emission_nm_col] = pd.to_numeric(df[emission_nm_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

if df[emission_nm_col].isna().any():
    raise ValueError("Emission column has blank or non-numeric values.")

if df[target_col].isna().any():
    raise ValueError("FWHM target column has blank or non-numeric values.")

if (df[emission_nm_col] <= 0).any():
    raise ValueError("Emission wavelength must be positive.")

X = df[feature_cols].copy()
y = df[target_col].copy()

X = X.apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Feature table has blank or non-numeric values. Please check the Excel file.")

print("Shape of X:", X.shape)
print("Number of features:", X.shape[1])
print("Feature columns:", feature_cols)


def fwhm_ev_to_nm(emission_nm, fwhm_ev):
    emission_nm = np.asarray(emission_nm, dtype=float)
    fwhm_ev = np.asarray(fwhm_ev, dtype=float)

    e0 = HC / emission_nm

    lambda_low = HC / (e0 + fwhm_ev / 2.0)
    lambda_high = HC / (e0 - fwhm_ev / 2.0)

    return lambda_high - lambda_low


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

kf = KFold(n_splits=10, shuffle=True, random_state=15)
pred_eV = np.zeros(len(df))

for train_idx, val_idx in kf.split(X):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

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


actual_nm = fwhm_ev_to_nm(df[emission_nm_col].values, y.values)
pred_nm = fwhm_ev_to_nm(df[emission_nm_col].values, pred_eV)

r2_nm = r2_score(actual_nm, pred_nm)
mae_nm = mean_absolute_error(actual_nm, pred_nm)
rmse_nm = np.sqrt(mean_squared_error(actual_nm, pred_nm))

print("\n10-fold CV performance in nm")
print(f"R2   = {r2_nm:.2g}")
print(f"MAE  = {mae_nm:.2g} nm")
print(f"RMSE = {rmse_nm:.2g} nm")


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

plt.xlabel("Actual FWHM (nm)", fontsize=11)
plt.ylabel("Predicted FWHM (nm)", fontsize=11)
plt.title("CatBoost 10-Fold CV Parity Plot", fontsize=12)

plt.text(
    0.05,
    0.95,
    f"R² = {r2_nm:.2g}\nMAE = {mae_nm:.2g} nm\nRMSE = {rmse_nm:.2g} nm",
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
predict_file = Path.cwd() / "To_predict_fwhm.xlsx"  # Keep this prediction file in the same folder
output_file = Path.cwd() / "predicted_fwhm.xlsx"    # Output will be saved in the same folder

predict_df = pd.read_excel(predict_file)
predict_df.columns = predict_df.columns.astype(str).str.strip()

# Features start from column B in prediction file
X_new = predict_df.iloc[:, 1:].copy()
X_new = X_new.apply(pd.to_numeric, errors="coerce")
X_new = X_new.fillna(X.median(numeric_only=True))

if X_new.shape[1] != X.shape[1]:
    raise ValueError(
        f"Feature mismatch: training has {X.shape[1]} features, "
        f"prediction file has {X_new.shape[1]} features."
    )

X_new.columns = feature_cols

# Train final model on the full training set
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
X_new_scaled = final_scaler.transform(X_new)

final_model = CatBoostRegressor(**params)
final_model.fit(X_scaled, y)

pred_fwhm_eV = final_model.predict(X_new_scaled)

result_df = predict_df.copy()
result_df["prediction_FWHM_eV"] = pred_fwhm_eV

save_cols = ["Formula", "prediction_FWHM_eV", "x"]

missing_cols = [c for c in save_cols if c not in result_df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in prediction output: {missing_cols}")

result_df[save_cols].to_excel(output_file, index=False)

print("\nTop 10 predicted FWHM results:")
print(result_df[save_cols].head(10).to_string(index=False))

print(f"\nPredictions saved to: {output_file}")

