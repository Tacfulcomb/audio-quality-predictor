import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load the results from batch evaluation
y_true = np.load("sota_true.npy")
y_pred = np.load("sota_preds.npy")

# 2. Create a professional-grade plot
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# Scatter plot with some transparency (alpha) to show density
plt.scatter(y_true, y_pred, alpha=0.4, color='#1f77b4', label='Predicted vs Actual')

# The 'Perfect Prediction' line
line_coords = [y_true.min(), y_true.max()]
plt.plot(line_coords, line_coords, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

# Adding text box with metrics
stats_text = f'R² Score: {r2_score(y_true, y_pred):.2f}\nMSE: {mean_squared_error(y_true, y_pred):.2f}'
plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.title('Deep Learning SOTA: Predicted vs. Actual MOS', fontsize=15)
plt.xlabel('Ground Truth MOS (Human Rating)', fontsize=12)
plt.ylabel('Predicted MOS (UTMOS22)', fontsize=12)
plt.legend(loc='lower right')
plt.tight_layout()

# Save the asset
plt.savefig('sota_results_plot.png', dpi=300)
print("Asset saved: sota_results_plot.png")
plt.show()