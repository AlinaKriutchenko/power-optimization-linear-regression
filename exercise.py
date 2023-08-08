import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the first Feather file (machine1.feather)
data_machine1 = pd.read_feather('machine1.feather')

# Read the second Feather file (machine2.feather)
data_machine2 = pd.read_feather('machine2.feather')

# Print data
print(data_machine1)
print(data_machine2)

# Filter data for machine type #1
data_machine1_filtered = data_machine1[(data_machine1['check'] >= 90) & (data_machine1['check'] <= 110)]

# Filter data for machine type #2
data_machine2_filtered = data_machine2[(data_machine2['check'] >= 90) & (data_machine2['check'] <= 110)]


def train_model(data, machine_type):
    X = data[['input_1', 'input_2', 'input_3']]
    y = data['power']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict power usage for the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error as a metric of accuracy
    mse = mean_squared_error(y_test, y_pred)

    # Plot the results
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Power')
    plt.ylabel('Predicted Power')
    plt.title(f'Machine Type #{machine_type} - Power Prediction (MSE: {mse:.2f})')
    plt.show()

    return model, mse


# Train model for machine type #1
model_machine1, mse_machine1 = train_model(data_machine1_filtered, 1)

# Train model for machine type #2
model_machine2, mse_machine2 = train_model(data_machine2_filtered, 2)

# Print the MSE for each machine type
print(f'MSE for Machine Type #1: {mse_machine1:.2f}')
print(f'MSE for Machine Type #2: {mse_machine2:.2f}')


# Define the target total GPH for the factory
target_total_GPH = 9000

# Define the GPH range for each machine type
GPH_range_machine1 = range(400, 426, 1)  # Reducing the range from 180-600 to 400-425 with a step of 1
GPH_range_machine2 = range(625, 676, 1)  # Reducing the range from 300-1000 to 625-675 with a step of 1


# Function to calculate total power for a given GPH configuration
def calculate_total_power(model1, model2, GPH_values_machine1, GPH_values_machine2):
    # Convert numpy arrays to DataFrames with correct column names
    df_machine1 = pd.DataFrame({'input_1': 25, 'input_2': 6, 'input_3': GPH_values_machine1})
    df_machine2 = pd.DataFrame({'input_1': 25, 'input_2': 6, 'input_3': GPH_values_machine2})

    # Calculate power for each machine using the trained models
    power_machine1 = model1.predict(df_machine1)
    power_machine2 = model2.predict(df_machine2)

    # Calculate total power for all machines
    total_power = sum(power_machine1) + sum(power_machine2)

    return total_power

# Initialize variables to store the optimal GPH values and total power
optimal_GPH_machine1 = 0
optimal_GPH_machine2 = 0
lowest_total_power = float('inf')

# Iterates through all possible GPH values for each machine type and find the optimal configuration
for GPH_machine1 in range(180, 601):
    for GPH_machine2 in range(300, 1001):
        # Introduce randomness to initial GPH values for each machine type
        GPH_machine1_initial = GPH_machine1 + random.randint(-20, 20)  # Adding random value between -20 and 20
        GPH_machine2_initial = GPH_machine2 + random.randint(-20, 20)  # Adding random value between -20 and 20
        
        total_power = calculate_total_power(model_machine1, model_machine2, [GPH_machine1_initial] * 10, [GPH_machine2_initial] * 10)
        if total_power < lowest_total_power and (GPH_machine1_initial * 10 + GPH_machine2_initial * 10) == target_total_GPH:
            lowest_total_power = total_power
            optimal_GPH_machine1 = GPH_machine1_initial
            optimal_GPH_machine2 = GPH_machine2_initial

# Prints the optimal GPH values and total power
print(f'Optimal GPH for Machine Type #1: {optimal_GPH_machine1}')
print(f'Optimal GPH for Machine Type #2: {optimal_GPH_machine2}')
print(f'Total Power: {lowest_total_power:.2f}')


# Optimization evaluation. Calculate total power consumption before optimization
total_power_before_optimization = calculate_total_power(model_machine1, model_machine2, [400] * 10, [625] * 10)

# Calculate total power consumption after optimization
total_power_after_optimization = calculate_total_power(model_machine1, model_machine2, [optimal_GPH_machine1] * 10, [optimal_GPH_machine2] * 10)

# Print the results
print(f'Total Power Consumption Before Optimization: {total_power_before_optimization:.2f}')
print(f'Total Power Consumption After Optimization: {total_power_after_optimization:.2f}')

# Calculate power savings
power_savings = total_power_before_optimization - total_power_after_optimization
print(f'Power Savings: {power_savings:.2f}')


# Save evaluation results to output file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Machine type #1 - MSE: {mse_machine1:.2f}\n")
    f.write(f"Machine type #2 - MSE: {mse_machine2:.2f}\n")

# Save optimization results to output file
with open('optimization_results.txt', 'w') as f:
    for i in range(20):
        f.write(f"Machine {i+1} - GPH: {optimal_GPH_machine1 if i < 10 else optimal_GPH_machine2}\n")
    total_power_consumption = calculate_total_power(model_machine1, model_machine2, [optimal_GPH_machine1] * 10, [optimal_GPH_machine2] * 10)
    f.write(f"Total Power Consumption: {total_power_consumption:.2f}\n")
