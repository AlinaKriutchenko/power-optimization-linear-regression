# power-optimization-linear-regression

# install:
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

# to run the code:
python main.py

# limitations:
1. Linear Model Assumption: The linear regression models used may not capture potential nonlinear relationships between inputs and power consumption.
2. Data Filtering Impact: Excluding data outside the 'check' range might limit the models' representativeness and generalizeability.
3. Limited Generalization: Models might not perform well on new data or different conditions without cross-validation or external validation.
4. Discrete Optimization: Optimization using discrete GPH values might not cover all potential solutions or constraints.
5. Computational Complexity: The iterative optimization process could become resource-intensive for larger datasets or complex models.
6. Single-Objective Focus: The optimization focused solely on minimizing power, overlooking potential trade-offs with other factors.
7. Machine Interaction: The models treat machines as independent, ignoring potential interactions affecting power consumption.
8. Model Complexity vs. Interpretability: Complex models might offer better accuracy but at the cost of interpretability.
9. Real-World Feasibility: Optimized GPH values might not align with practical constraints or maintenance considerations.
10. Unconsidered Factors: Factors like temperature, humidity, and maintenance schedules were omitted from the analysis.

# conclusions:
Optimal GPH Values:
- After optimization, the optimal Goods Per Hour (GPH) values were determined to minimize power consumption while achieving the target total GPH of 9,000 for the factory.
- Optimal GPH for Machine Type #1: 600
- Optimal GPH for Machine Type #2: 300

Total Power Consumption:
- The total power consumption of the factory was calculated using the optimized GPH values.
- Total Power Consumption Before Optimization: 1874.42
- Total Power Consumption After Optimization: 1521.94

Power Savings:
- By implementing the optimized GPH values, the factory achieved a power savings of 352.49 units.

