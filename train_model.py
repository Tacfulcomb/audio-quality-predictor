import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. FEATURE DEFINITIONS  ---
feature_names = (
    [f'MFCC_{i}_Mean' for i in range(1, 14)] + 
    [f'MFCC_{i}_Std' for i in range(1, 14)] + 
    ['Flatness_Mean', 'Flatness_Std', 
     'Rolloff_Mean', 'Rolloff_Std', 
     'Centroid_Mean', 'ZCR_Mean']
)

# --- 2. LOAD & SPLIT DATA ---
print(" Loading Level 2 Perceptual Data...")
X = np.load("X_features_v2.npy")
y = np.load("y_labels_v2.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. TRAIN ENHANCED MODEL ---
print(f" Training Random Forest on {len(X_train)} samples with 32 features...")
model = RandomForestRegressor(
    n_estimators=500, 
    max_depth=15, 
    min_samples_leaf=2, 
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "improved_ml_model.pkl")
print(" Model saved as 'improved_ml_model.pkl'")

# --- 4. EVALUATE METRICS ---
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
pearson_r = np.corrcoef(y_test, predictions)[0, 1]

print("\n" + "="*35)
print(" Level 2: Perceptual Baseline Results ")
print("="*35)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score:    {r2:.4f}")
print(f"Pearson R:          {pearson_r:.4f}")
print("="*35)

# --- 5. VISUALIZATION 1: PERFORMANCE SCATTER ---
def plot_level2_performance(y_test, preds, r2_val):
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    plt.scatter(y_test, preds, alpha=0.5, color='#2ecc71', label='Level 2 Predictions')
    plt.plot([1, 5], [1, 5], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    stats = f'R² Score: {r2_val:.4f}\nPearson r: {pearson_r:.4f}'
    plt.gca().text(0.05, 0.95, stats, transform=plt.gca().transAxes, 
                 fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.title('Level 2: Perceptual ML Performance (32 Features)', fontsize=15)
    plt.xlabel('Ground Truth MOS (Human)', fontsize=12)
    plt.ylabel('Predicted MOS (Random Forest)', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig('level2_performance.png', dpi=300, bbox_inches='tight')
    print(" Saved performance plot: 'level2_performance.png'")
    plt.close()

# --- 6. VISUALIZATION 2: FEATURE IMPORTANCE ---
def plot_feature_importance(model, feat_names):
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Take the top 15 features 
    top_n = 15
    sorted_features = [feat_names[i] for i in indices][:top_n]
    sorted_importances = importances[indices][:top_n]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    
    plt.title("Level 2: Top 15 Most Important Perceptual Features", fontsize=16)
    plt.xlabel("Random Forest Importance Score (Gini)", fontsize=12)
    plt.ylabel("Acoustic Descriptors", fontsize=12)
    plt.tight_layout()
    plt.savefig('level2_feature_importance.png', dpi=300)
    print(" Saved importance plot: 'level2_feature_importance.png'")
    plt.close()

# Run the plotters
plot_level2_performance(y_test, predictions, r2)
plot_feature_importance(model, feature_names)