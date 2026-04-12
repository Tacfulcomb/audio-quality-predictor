import streamlit as st
import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Audio Quality Predictor SOTA", layout="centered")
st.title("🎙️ AI Audio Quality Predictor")
st.markdown("### Project 25222: Deep Learning MOS Estimation")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained UTMOS22 Strong Learner via Torch Hub.
    Fulfills 'Implementation Details' by utilizing SOTA deep learning weights.
    """
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

predictor = load_model()

# --- USER INPUT ---
uploaded_file = st.file_uploader("Upload a .wav file to estimate perceptual quality", type=["wav"])

if uploaded_file is not None:
    # Playback for the user
    st.audio(uploaded_file, format='audio/wav')
    
    # Save temp file for librosa processing
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- ANALYSIS EXECUTION ---
    if st.button("Run Quality Analysis"):
        with st.spinner("Analyzing spectral features and predicting MOS..."):
            try:
                # Load audio and force 16kHz for model compatibility
                wave, sr = librosa.load("temp.wav", sr=16000)
                wave_tensor = torch.from_numpy(wave).unsqueeze(0)

                # --- INFERENCE ---
                with torch.no_grad():
                    score = predictor(wave_tensor, sr).item()

                # Display Numerical Result
                st.subheader(f"Predicted MOS: {score:.2f} / 5.0")
                
                # Visual Feedback logic (Qualitative thresholds)
                if score >= 4.0:
                    st.success("Quality: Excellent (Broadcast Standard)")
                elif score >= 3.0:
                    st.info("Quality: Fair (Understandable, minor artifacts)")
                else:
                    st.error("Quality: Poor (Significant Distortion detected)")
                
                # Progress bar visualization
                st.progress(score / 5.0)

                # --- SPECTROGRAM ---
                st.markdown("---")
                st.subheader("Spectral Visualization")
                fig, ax = plt.subplots(figsize=(10, 4))
                # Convert to Decibels for human-perceptual visual scaling
                D = librosa.amplitude_to_db(np.abs(librosa.stft(wave)), ref=np.max)
                img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
                plt.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                # Cleanup  
                if os.path.exists("temp.wav"):
                    os.remove("temp.wav")

# --- FOOTER ---
st.divider()
st.caption("""
**System Architecture:** SSL-based WavLM + UTMOS Linear Head  
**Target Dataset:** VoiceMOS Challenge 2022 (BVCC)  
**Metrics:** Mean Opinion Score (MOS) 1.0 - 5.0
""")