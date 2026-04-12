import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score

# Load model
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

# Load your 'laid bare' ground truth from Week 1
# Ensure the path matches your actual VoiceMos folder
df = pd.read_csv("VoiceMos/main/DATA/sets/train_mos_list.txt", header=None, names=['wav_name', 'mos'])
df['path'] = df['wav_name'].apply(lambda x: f"VoiceMos/main/DATA/wav/{x}")

# Filter to only the 226 files you actually have
import os
df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)

y_true = []
y_pred = []

print(f"Evaluating SOTA model on {len(df)} files...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        wave, sr = librosa.load(row['path'], sr=16000)
        wave_tensor = torch.from_numpy(wave).unsqueeze(0)
        
        with torch.no_grad():
            score = predictor(wave_tensor, sr).item()
            
        y_true.append(row['mos'])
        y_pred.append(score)
    except Exception as e:
        continue

# 3. Final Calculations for Requirement 7
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\n--- SOTA Evaluation Results ---")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f} (This should be much higher!)")

# Save predictions for your report graphs
np.save("sota_preds.npy", y_pred)
np.save("sota_true.npy", y_true)