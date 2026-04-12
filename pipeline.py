import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

class AudioQualityPipeline:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def extract_features(self, file_path):
        """
        Extracts an expanded handcrafted feature set approximating eGeMAPS.
        Fulfills Requirement 1 (Methodology) for handcrafted ML features.
        """
        try:
            y, sr = librosa.load(file_path, sr=self.target_sr)
            
            # --- 1. MFCCs (The standard 'texture' features) ---
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_std = np.std(mfccs.T, axis=0)
            
            # --- 2. Spectral Features (Capturing 'Dullness' and 'Smearing') ---
            # Spectral Flatness: Measures if the sound is noise-like vs tone-like
            flatness = librosa.feature.spectral_flatness(y=y)
            flatness_mean = np.mean(flatness)
            flatness_std = np.std(flatness)
            
            # Spectral Rolloff: Measures the bandwidth (identifies low-pass filters)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rolloff_mean = np.mean(rolloff)
            rolloff_std = np.std(rolloff)
            
            # Spectral Centroid: Indicates the 'brightness' of the audio
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            centroid_mean = np.mean(centroid)
            
            # --- 3. Temporal Features (Capturing 'Dynamics') ---
            # Zero Crossing Rate: Detects high-frequency noise/hiss
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            
            # Combine all (26 MFCCs + 5 Perceptual = 31 total features)
            combined = np.hstack([
                mfccs_mean, mfccs_std, 
                flatness_mean, flatness_std, 
                rolloff_mean, rolloff_std, 
                centroid_mean, zcr_mean
            ])
            return combined
        except Exception as e:
            print(f"Error: {e}")
            return None

def load_voicemos_data(base_path, list_file):
    df = pd.read_csv(list_file, header=None, names=['wav_name', 'mos'])
    df['path'] = df['wav_name'].apply(lambda x: os.path.join(base_path, "wav", x))
    return df[['path', 'mos']]

def run_extraction_batch(df, pipeline):
    X, y = [], []
    print(f"Extracting perceptual features for {len(df)} files...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.exists(row['path']):
            features = pipeline.extract_features(row['path'])
            if features is not None:
                X.append(features)
                y.append(row['mos'])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    pipe = AudioQualityPipeline()
    voicemos_base = os.path.join("VoiceMos", "main", "DATA")
    voicemos_list = os.path.join(voicemos_base, "sets", "train_mos_list.txt")
    
    if os.path.exists(voicemos_list):
        df_all = load_voicemos_data(voicemos_base, voicemos_list)
        # SPRINT UPDATE: Take all 2254 samples for maximum accuracy
        X, y = run_extraction_batch(df_all, pipe)
        
        if len(X) > 0:
            np.save("X_features_v2.npy", X)
            np.save("y_labels_v2.npy", y)
            print(f"\nSuccess! Extracted {len(X)} samples with perceptual features.")