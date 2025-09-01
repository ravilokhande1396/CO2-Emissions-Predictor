import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr  

# 1. Load Dataset
data = pd.read_csv("FuelConsumption.csv")
print("Dataset shape:", data.shape)
print(data.head())


# 2. Select relevant features 
features = ["ENGINESIZE", "CYLINDERS","FUELCONSUMPTION_CITY",]
target = "CO2EMISSIONS"

X = data[features]
y = data[target]

X = data[features]
y = data[target]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Pearson correlation
r, _ = pearsonr(y_test, y_pred)

# 7. Visualization - Parity Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6, label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],color="red", linestyle="--", linewidth=2, label="Perfect Prediction (Parity Line)")

plt.xlabel("Actual CO2 Emissions", fontsize=14, fontweight="bold")
plt.ylabel("Predicted CO2 Emissions", fontsize=14, fontweight="bold")
plt.title("Parity Plot: Predicted vs Actual CO2 Emissions", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.axis("equal")
plt.grid(True, linestyle="--", alpha=0.6)

# Add metrics in top-left corner
metrics_text = f"R² = {r2:.3f}\nPearson r = {r:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}"
plt.gca().text(0.05, 0.95, metrics_text,fontsize=12, color="black", fontweight="bold",verticalalignment="top", horizontalalignment="left",
               transform=plt.gca().transAxes,bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5", alpha=0.7))
plt.tight_layout()
plt.savefig("parity_plot.png", dpi=300) 
plt.show()

# 8. Visualization - Engine Size vs CO2 (Actual vs Predicted)
plt.figure(figsize=(8,6))
plt.scatter(X_test["ENGINESIZE"], y_test, color="red", alpha=0.6, label="Actual")
plt.scatter(X_test["ENGINESIZE"], y_pred, color="blue", alpha=0.6, label="Predicted")

plt.xlabel("Engine Size", fontsize=14, fontweight="bold")
plt.ylabel("CO2 Emissions", fontsize=14, fontweight="bold")
plt.title("Engine Size vs CO2 Emissions (Actual vs Predicted)", fontsize=16, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("engine_size_vs_co2.png", dpi=300)  
plt.show()
