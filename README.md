# CO2 Emissions Predictor

Predict vehicle CO₂ emissions based on engine specifications and fuel consumption using Linear Regression and Random Forest Regression.

#Project Overview
This project demonstrates how to build, evaluate, and interpret machine learning models for predicting CO₂ emissions of vehicles using the Canadian Fuel Consumption dataset from Kaggle.

Install the following libraries : 
pip install pandas, numpy, matplotlib, scikit-learn, scipy.

Before running,  make a main folder and add the following files, and create a "plots" folder inside the main folder:

linear_regression_fuel.py

random_forest_fuel.py

FuelConsumption.csv

plots/
   parity_plot_linear.png
   
   engine_size_vs_co2.png
   
   parity_plot_rf.png
   
   feature_importance.png   (saved files after run)

Run the scripts:

linear_regression_fuel.py → Linear Regression results

random_forest_fuel.py → Random Forest results & feature importance

Results

Linear Regression: R² ≈ 0.88, MAE ≈ 16.7 g/km, RMSE ≈ 22.6 g/km

Random Forest: R² ≈ 0.92+, lower error, captures nonlinear patterns

Feature importance shows that Fuel Consumption City and Engine Size are the strongest predictors.





