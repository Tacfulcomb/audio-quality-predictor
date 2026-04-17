import librosa
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. CONFIGURATION ---
BASE_PATH = os.path.join("VoiceMos", "main", "DATA")
LIST_FILE = os.path.join(BASE_PATH, "sets", "train_mos_list.txt")
TARGET_SR = 16000

def extract_lvl1_features(file_path):
    """Level 1: Strictly 13 MFCCs (Static Baseline)"""
    try:
        y, _ = librosa.load(file_path, sr=TARGET_SR)
        mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
        return np.mean(mfccs.T, axis=0) # 13-dim vector
    except Exception as e:
        return None

# --- 2. DATA PREPARATION ---
print("🎙️ Loading Data for Level 1 Evaluation...")
df = pd.read_csv(LIST_FILE, header=None, names=['wav_name', 'mos'])
X_list, y_list = [], []

print("Extracting Baseline 13-MFCC Features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    p = os.path.join(BASE_PATH, "wav", row['wav_name'])
    feat = extract_lvl1_features(p)
    if feat is not None:
        X_list.append(feat)
        y_list.append(row['mos'])

X = np.array(X_list)
y = np.array(y_list)

# Use the exact same split as training to test on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. MODEL INFERENCE ---
print("\n🤖 Loading Level 1 Model...")
model_l1 = joblib.load('audio_quality_model.pkl')

predictions = model_l1.predict(X_test)

# --- 4. METRICS CALCULATION ---
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
pearson_r = np.corrcoef(y_test, predictions)[0, 1]

print("\n" + "="*30)
print(" Level 1: Statistical Baseline Results ")
print("="*30)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score:    {r2:.4f}")
print(f"Pearson R:          {pearson_r:.4f}")
print("="*30)

# --- 5. VISUALIZATION ---
def plot_level1_results(y_test, preds, r2_val):
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Plotting
    plt.scatter(y_test, preds, alpha=0.5, color='#3498db', label='Level 1 Predictions')
    plt.plot([1, 5], [1, 5], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Add stats box
    stats = f'R² Score: {r2_val:.4f}'
    plt.gca().text(0.05, 0.95, stats, transform=plt.gca().transAxes, 
                 fontsize=14, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Titles and Labels
    plt.title('Level 1: Statistical Baseline Performance (13 MFCCs)', fontsize=15)
    plt.xlabel('Ground Truth MOS (Human)', fontsize=12)
    plt.ylabel('Predicted MOS (Random Forest)', fontsize=12)
    plt.legend(loc='lower right')
    
    # Save for the report
    plt.savefig('level1_baseline_performance.png', dpi=300)
    print("\n✅ Plot saved as 'level1_baseline_performance.png'")
    plt.show()

plot_level1_results(y_test, predictions, r2)