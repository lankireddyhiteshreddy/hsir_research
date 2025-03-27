import numpy as np
from scipy.stats import pearsonr

# Compute Correlation Coefficient (CC)
correlation_coefficient, _ = pearsonr(y_test, y_pred)

# Compute Relative Absolute Error (RAE)
rae = (np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test - np.mean(y_test)))) * 100

# Print results
print(f"Correlation Coefficient (CC): {correlation_coefficient:.4f}")
print(f"Relative Absolute Error (RAE): {rae:.4f} %")
