import streamlit as st
import joblib
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="HUST Audio Quality Lab", layout="wide", page_icon="🎙️")

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_all_models():
    try:
        lvl1 = joblib.load("audio_quality_model.pkl")
        lvl2 = joblib.load("improved_ml_model.pkl")
        lvl3_bundle = joblib.load("informed_hybrid_lvl3.joblib")
        lvl4 = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        return lvl1, lvl2, lvl3_bundle, lvl4
    except Exception as e:
        st.error(f"Error loading models: {e}. Check if file names match exactly.")
        return None, None, None, None

models = load_all_models()

# --- 4. FEATURE ENGINE ---
def get_features(y, sr, level, system_id=None, system_map=None):
    # --- SHARED BASE: MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # --- LEVEL 1: Statistical Baseline (13 features) ---
    if level == "Level 1":
        return mfccs_mean.reshape(1, -1)

    # --- LEVEL 2: Perceptual Baseline (The 32-Feature Recipe from pipeline.py) ---
    if level == "Level 2":
        mfccs_std = np.std(mfccs.T, axis=0)
        flatness = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # This matches the 32-feature requirement
        l2_vector = np.hstack([
            mfccs_mean, mfccs_std, 
            np.mean(flatness), np.std(flatness), 
            np.mean(rolloff), np.std(rolloff), 
            np.mean(centroid), np.mean(zcr)
        ])
        return l2_vector.reshape(1, -1)

    # --- LEVEL 3: Informed Hybrid (The 28/29-Feature Recipe) ---
    if level == "Level 3":
        deltas_mean = np.mean(librosa.feature.delta(mfccs).T, axis=0)
        flatness_val = np.mean(librosa.feature.spectral_flatness(y=y))
        centroid_val = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        interaction = flatness_val * centroid_val
        
        # 28 Acoustic features
        acoustic_vec = np.hstack([mfccs_mean, deltas_mean, flatness_val, interaction])
        
        # +1 Metadata Prior = 29 Total
        prior = system_map.get(system_id, 3.0) 
        return np.append(acoustic_vec, prior).reshape(1, -1)

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.divider()
    selected_level = st.radio("Hierarchy Depth", ["Level 1", "Level 2", "Level 3", "Level 4"])
    
    system_choice = None
    if selected_level == "Level 3":
        st.info("Level 3 requires a System ID to 'Inform' the prediction.")
        system_choice = st.selectbox("Synthesis System ID", options=list(models[2]['system_map'].keys()))
    
    st.divider()
    st.caption("Project 25222: Deep Learning MOS Estimation")

# --- 6. MAIN INTERFACE ---
st.title("🎙️ AI Audio Quality Predictor")
st.markdown("### Hierarchical MOS Estimation for Synthetic Speech")

# File Uploader
uploaded_file = st.file_uploader("Upload a .wav audio sample", type=["wav"])

if uploaded_file:
    # Playback section
    col_audio, col_meta = st.columns([2, 1])
    with col_audio:
        st.audio(uploaded_file, format='audio/wav')
    
    # Save temp file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button(" Run Prediction Analysis", use_container_width=True):
        with st.spinner("Extracting features and running inference..."):
            y, sr = librosa.load("temp.wav", sr=16000)
            
            # --- MODEL INFERENCE ---
            score = 0.0
            if selected_level == "Level 1":
                feat = get_features(y, sr, "Level 1")
                score = models[0].predict(feat)[0]
            elif selected_level == "Level 2":
                feat = get_features(y, sr, "Level 2")
                score = models[1].predict(feat)[0]
            elif selected_level == "Level 3":
                feat = get_features(y, sr, "Level 3", system_choice, models[2]['system_map'])
                score = models[2]['model'].predict(feat)[0]
            elif selected_level == "Level 4":
                wave_tensor = torch.from_numpy(y).unsqueeze(0)
                with torch.no_grad():
                    score = models[3](wave_tensor, sr).item()

            # --- DISPLAY RESULTS ---
            st.divider()
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric(label=f"Predicted MOS ({selected_level})", value=f"{score:.2f} / 5.0")
                if score >= 4.0: st.success("Quality: Excellent")
                elif score >= 3.0: st.info("Quality: Fair")
                else: st.error("Quality: Poor")

                st.progress(float(score) / 5.0)

            with c2:
                # --- SPECTROGRAM DISPLAY ---
                fig, ax = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='magma')
                plt.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set_title("Spectral Power Density (Log-Frequency)")
                st.pyplot(fig)

    # Cleanup
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
else:
    st.info("Please upload a .wav file to begin analysis.")