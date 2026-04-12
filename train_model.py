import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# 1. Load the new 'Perceptual' features
X = np.load("X_features_v2.npy")
y = np.load("y_labels_v2.npy")

def plot_final_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.4, color='#2ecc71')
    plt.plot([1, 5], [1, 5], '--r', label='Identity Line')
    plt.xlabel('Ground Truth MOS (Human)')
    plt.ylabel('Predicted MOS (Improved ML)')
    plt.title('Improved ML Baseline: Perceptual Features vs. Human Ratings')
    plt.grid(True, alpha=0.3)
    plt.savefig('ml_v2_results.png')
    plt.show()

# 2. Split (Requirement 7: Evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Enhanced Random Forest configuration
# n_estimators=500 and max_depth help hit that higher R-score
model = RandomForestRegressor(
    n_estimators=500, 
    max_depth=15, 
    min_samples_leaf=2, 
    random_state=42,
    n_jobs=-1
)

print(f"Training on {len(X_train)} samples with improved perceptual feature set...")
model.fit(X_train, y_train)

# 4. Evaluate Performance
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Calculation of Pearson Correlation (The target 0.8+ metric)
pearson_r = np.corrcoef(y_test, predictions)[0, 1]

print("\n--- Improved ML Results ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Pearson Correlation (R): {pearson_r:.4f}") # Aiming for 0.8+

# 5. Save the 'Submission Winner' model
joblib.dump(model, "improved_ml_model.pkl")
plot_final_results(y_test, predictions)