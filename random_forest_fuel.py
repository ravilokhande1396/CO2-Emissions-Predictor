import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# 1. Load Dataset
data = pd.read_csv("FuelConsumption.csv")
print("Dataset shape:", data.shape)

# 2. Features & Target
features = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY"]
target = "CO2EMISSIONS"

X = data[features]
y = data[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# 5. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
r, _ = pearsonr(y_test, y_pred)

print("Random Forest Performance:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
print(f"Pearson r: {r:.3f}")

# 6. Parity Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="green", alpha=0.6, label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual CO2 Emissions", fontsize=14, fontweight="bold")
plt.ylabel("Predicted CO2 Emissions", fontsize=14, fontweight="bold")
plt.title("Random Forest: Parity Plot", fontsize=16, fontweight="bold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

metrics_text = f"R² = {r2:.3f}\nPearson r = {r:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}"
plt.gca().text(0.05, 0.95, metrics_text, fontsize=12, color="black", fontweight="bold",
               verticalalignment="top", horizontalalignment="left",
               transform=plt.gca().transAxes,
               bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5", alpha=0.7))

plt.tight_layout()
plt.savefig("plots/parity_plot_rf.png", dpi=300)
plt.show()

# 7. Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,6))
plt.bar(range(len(features)), importances[indices], color="skyblue")
plt.xticks(range(len(features)), [features[i] for i in indices], fontsize=12, fontweight="bold")
plt.ylabel("Feature Importance", fontsize=14, fontweight="bold")
plt.title("Random Forest Feature Importance", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=300)
plt.show()
