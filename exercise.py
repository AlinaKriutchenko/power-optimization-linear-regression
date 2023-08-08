import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize  # Import the 'minimize' function


import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Load the datasets for machine type #1 and machine type #2
data_machine1 = pd.read_feather('machine1.feather')
data_machine2 = pd.read_feather('machine2.feather')

# Filter out rows with 'check' value between 90 and 110
data_machine1 = data_machine1[(data_machine1['check'] >= 90) & (data_machine1['check'] <= 110)]
data_machine2 = data_machine2[(data_machine2['check'] >= 90) & (data_machine2['check'] <= 110)]

# Split the data into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(data_machine1[['input_1', 'input_2', 'input_3']], data_machine1['power'], test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_machine2[['input_1', 'input_2', 'input_3']], data_machine2['power'], test_size=0.2, random_state=42)

# Create and train models for each machine type
model1 = LinearRegression()
model1.fit(X_train1, y_train1)

model2 = RandomForestRegressor()
model2.fit(X_train2, y_train2)

# Predict power for the test set
y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)

# Calculate evaluation metrics
mae1 = mean_absolute_error(y_test1, y_pred1)
mae2 = mean_absolute_error(y_test2, y_pred2)

mse1 = mean_squared_error(y_test1, y_pred1)
mse2 = mean_squared_error(y_test2, y_pred2)

print(f"Machine type #1 - MAE: {mae1}, MSE: {mse1}")
print(f"Machine type #2 - MAE: {mae2}, MSE: {mse2}")

# Define Objective Function for Optimization
def total_power(gph_values):
    # Calculate power consumption for each machine based on GPH values and models
    power1 = model1.predict([[25, 6, gph] for gph in gph_values[:10]])
    power2 = model2.predict([[25, 6, gph] for gph in gph_values[10:]])

    # Calculate total power consumption of the factory
    total_power = sum(power1) + sum(power2)
    return total_power

# Set GPH limits for each machine type
gph_limits = [(180, 600)] * 10 + [(300, 1000)] * 10

# Set target total GPH for the factory
target_total_gph = 9000

# Define the objective function for optimization
def objective_function(gph_values):
    return total_power(gph_values)

# Perform optimization
result = minimize(objective_function, [300] * 20, bounds=gph_limits, constraints={'type': 'eq', 'fun': lambda x: sum(x) - target_total_gph})
optimal_gph_values = result.x

# Save evaluation results to output file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Machine type #1 - MAE: {mae1}, MSE: {mse1}\n")
    f.write(f"Machine type #2 - MAE: {mae2}, MSE: {mse2}\n")

# Save optimization results to output file
with open('optimization_results.txt', 'w') as f:
    for i in range(20):
        f.write(f"Machine {i+1} - GPH: {optimal_gph_values[i]}\n")
    total_power_consumption = total_power(optimal_gph_values)
    f.write(f"Total Power Consumption: {total_power_consumption}\n")
