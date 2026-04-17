import librosa
import numpy as np
import pandas as pd
import os
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
BASE_PATH = "VoiceMos/main/DATA"
LIST_FILE = os.path.join(BASE_PATH, "sets", "train_mos_list.txt")
TARGET_SR = 16000

def extract_lvl1_features(file_path):
    try:
        y, _ = librosa.load(file_path, sr=TARGET_SR)
        # LEVEL 1: Only 13 MFCCs, nothing else.
        mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
        return np.mean(mfccs.T, axis=0) 
    except Exception as e:
        return None

# --- DATA PREP ---
df = pd.read_csv(LIST_FILE, header=None, names=['wav_name', 'mos'])
X_list, y_list = [], []

print(" Retraining Level 1: Extracting 13 MFCC features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    p = os.path.join(BASE_PATH, "wav", row['wav_name'])
    feat = extract_lvl1_features(p)
    if feat is not None:
        X_list.append(feat)
        y_list.append(row['mos'])

X = np.array(X_list)
y = np.array(y_list)

# --- TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Random Forest for the Baseline
model_l1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_l1.fit(X_train, y_train)

# --- SAVE & VERIFY ---
preds = model_l1.predict(X_test)
print(f"\n✅ Level 1 Retrained!")
print(f"R² Score: {r2_score(y_test, preds):.4f}")
print(f"Input Features: {X.shape[1]} (Should be 13)")

joblib.dump(model_l1, 'audio_quality_model.pkl')
print("File saved as 'audio_quality_model.pkl'")