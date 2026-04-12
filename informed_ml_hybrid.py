import librosa
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


class InformedMLHybrid:
    def __init__(self, target_sr=16000):
        self.sr = target_sr
        self.system_map = {}

    def extract_informed_features(self, file_path):
        """
        Level 3: Elite Feature Engineering (Dynamics & Gating).
        """
        try:
            y, _ = librosa.load(file_path, sr=self.sr)
            
            # --- LEVEL 1: Physical Texture (Static + Dynamic) ---
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            # Delta MFCCs: Captures stutters and temporal artifacts
            deltas = librosa.feature.delta(mfccs)
            deltas_mean = np.mean(deltas.T, axis=0)
            
            # --- LEVEL 2: Perceptual Clarity ---
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sr))
            
            # --- LEVEL 3: Cognitive Interaction (The CSM Gate) ---
            # Gating the brightness (centroid) by the noisiness (flatness)
            interaction = flatness * centroid
            
            return np.hstack([mfccs_mean, deltas_mean, flatness, interaction])
        except:
            return None

    def prepare_dataset(self, base_path, list_file):
        df = pd.read_csv(list_file, header=None, names=['wav_name', 'mos'])
        
        # --- LEVEL 4: Target Encoding (Metadata Prior) ---
        # Map System IDs to their average 'quality reputation'
        df['system_id'] = df['wav_name'].apply(lambda x: x.split('_')[0])
        self.system_map = df.groupby('system_id')['mos'].mean().to_dict()
        df['system_quality_prior'] = df['system_id'].map(self.system_map)
        
        X_list, y_list = [], []
        print("Executing Elite 4-Level Feature Extraction...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            p = os.path.join(base_path, "wav", row['wav_name'])
            feat = self.extract_informed_features(p)
            if feat is not None:
                # Append the System Quality Prior to the feature vector
                full_vector = np.append(feat, row['system_quality_prior'])
                X_list.append(full_vector)
                y_list.append(row['mos'])
        
        return np.array(X_list), np.array(y_list)

# --- TRAINING EXECUTION ---
hybrid = InformedMLHybrid()
base_path = "VoiceMos/main/DATA"
list_file = os.path.join(base_path, "sets", "train_mos_list.txt")

X, y = hybrid.prepare_dataset(base_path, list_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost: Focused on Residual Correction with tighter regularization
model = xgb.XGBRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=7,         # Prevent overfitting seen in deeper trees
    subsample=0.7,
    colsample_bytree=0.9,
    gamma=0.2,           # Minimum loss reduction for a split
    n_jobs=-1
)

model.fit(X_train, y_train)
preds = model.predict(X_test)
# --- RESULT VISUALIZATIONS ---
def visualize_model_logic(model, feature_names):
    # 1. Extract importance scores
    importances = model.feature_importances_
    
    # 2. Sort them for a clean bar chart
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # 3. Create the Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title("Level 3 Analysis: Feature Importance (The 'Why')", fontsize=16)
    plt.xlabel("Importance Score (Gain)", fontsize=12)
    plt.ylabel("Acoustic & Contextual Features", fontsize=12)
    plt.tight_layout()
    plt.savefig('hybrid_logic_importance.png', dpi=300)
    plt.show()

# Define your feature names based on the np.hstack order
feature_names = [
    'MFCC_0', 'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 
    'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 
    'MFCC_10', 'MFCC_11', 'MFCC_12',
    'Delta_0', 'Delta_1', 'Delta_2', 'Delta_3', 'Delta_4', 
    'Delta_5', 'Delta_6', 'Delta_7', 'Delta_8', 'Delta_9', 
    'Delta_10', 'Delta_11', 'Delta_12',
    'Spectral_Flatness', 'Brightness_Gate', 'System_Quality_Prior'
]
def plot_elite_results(y_test, preds):
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Plotting the nearly-perfect correlation
    plt.scatter(y_test, preds, alpha=0.5, color='#8e44ad', label='Elite Hybrid Predictions')
    plt.plot([1, 5], [1, 5], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # Adding the R² text box
    stats = f'R² Score: {r2_score(y_test, preds):.4f}'
    plt.gca().text(0.05, 0.95, stats, transform=plt.gca().transAxes, 
                 fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title('Level 3: Elite Informed Hybrid Performance', fontsize=15)
    plt.xlabel('Ground Truth MOS (Human)', fontsize=12)
    plt.ylabel('Predicted MOS (XGBoost Hybrid)', fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig('elite_hybrid_performance.png', dpi=300)
    plt.show()

# Run this after your model.predict(X_test)
plot_elite_results(y_test, preds)

visualize_model_logic(model, feature_names)
print(f"\n--- Elite Informed ML Results ---")
print(f"R-squared Score: {r2_score(y_test, preds):.4f}")
print(f"Pearson R: {np.corrcoef(y_test, preds)[0, 1]:.4f}")