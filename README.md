# power-optimization-linear-regression

# Install:
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

# To run the code:
python main.py


# Approach

**Model Choice**
Linear regression was selected due to its simplicity and interpretability. More complex models might provide better prediction, but for the purpose of understanding the relationship between inputs and power, linear regression suffices.

**Optimization Range**
The chosen GPH ranges (180-600 and 300-1000) respected the machines' capabilities. This avoided unrealistic or inefficient production rates while optimizing power consumption.

**Objective Function**
The main goal was to minimize power consumption while meeting the target GPH. Minimizing power aligns with energy conservation objectives, contributing to cost savings and reduced environmental impact.

**Metric Selection**
Mean Squared Error (MSE) was chosen to assess model accuracy. It quantifies the average squared difference between predicted and actual power, giving insight into how well the model predicts power values. MSE is a widely adopted metric in regression tasks.

**Iteration Approach**
The iterative approach was used to exhaustively search for optimal GPH values within the specified ranges. It aimed to find the configuration that minimizes power while meeting the production target.

**Evaluation Comparison**Comparing power consumption before and after optimization provided a tangible measure of the optimization's impact.


# Limitations:
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
11. Dataset size
