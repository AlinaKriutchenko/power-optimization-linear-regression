# power-optimization-linear-regression

## Install:
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

## To run the code:
python exercise.py


## Approach

**Model Choice:**
Linear regression was chosen due to its simplicity and interpretability.

**Optimization Range:**
Two GPH ranges were considered: 180-600 and 300-1000. These ranges were chosen to avoid unrealistic or inefficient production rates during optimization.

**Objective Function:**
The objective was to minimize power consumption, aligning with energy conservation goals.

**Metric Selection:**
Mean Squared Error (MSE) was selected as the primary metric, widely used for regression tasks. It quantifies the average squared difference between predicted and actual power values.

**Iteration Approach:**
An iterative approach was employed to exhaustively search for optimal GPH values. The goal was to minimize power consumption while achieving the production target.

**Evaluation Comparison:**
The impact of optimization was measured by comparing power consumption before and after the optimization process.


## Limitations:
1. Linear Model: May not capture potential nonlinear relationships between inputs and power consumption.
2. Data Filtering: Excluding data might limit the models' representativeness and generalizeability.
3. Limited Generalization: Models might not perform well on new data or different conditions without cross-validation or external validation.
4. Computational Complexity: The iterative optimization process could become resource-intensive for larger datasets or complex models.
5. Single-Objective Focus: The optimization focused solely on minimizing power, overlooking potential trade-offs with other factors.
6. Model Complexity vs. Interpretability: Complex models might offer better accuracy but at the cost of interpretability.
7. Real-World Feasibility: Optimized GPH values might not align with practical constraints or maintenance considerations.
8. Unconsidered Factors: Factors like temperature, humidity, and maintenance schedules were omitted from the analysis.


1. Linear Model: The linear model may not capture potential nonlinear relationships between inputs and power consumption.
2. Data Filtering: Excluding data could limit the representativeness and generalizability of the models.
3. Limited Generalization: Models might not perform well on new data or under different conditions without cross-validation or external validation.
4. Computational Complexity: The iterative optimization process could become resource-intensive for larger datasets or complex models.
5. Single-Objective Focus: The optimization focused solely on minimizing power, potentially overlooking trade-offs with other factors.
6. Model Complexity vs. Interpretability: Complex models might offer better accuracy but could sacrifice interpretability.
7. Real-World Feasibility: Optimized GPH values might not align with practical constraints or maintenance considerations.
8. Unconsidered Factors: Factors such as temperature, humidity, and maintenance schedules were omitted from the analysis.