# Perceptual Audio Quality Predictor
# AI Audio Quality Predictor: Hierarchical MOS Estimation for Synthetic Speech
**Project 25222** | **Hanoi University of Science and Technology (HUST)**
**Author:** Dinh Quang Hien (20224281)

## 📌 Project Overview
This repository contains the code and implementation for a 4-tier Hierarchical Mean Opinion Score (MOS) Estimation system. The goal of this project is to automate the quality assessment of Text-to-Speech (TTS) and Voice Conversion (VC) systems by modeling both acoustic physics and human cognitive bias (the "System Effect").

### The 4-Tier Hierarchy:
* **Level 1 (Static Baseline):** 13 MFCCs (Random Forest)
* **Level 2 (Perceptual ML):** eGeMAPS + Standard Deviations (Random Forest)
* **Level 3 (Informed Hybrid):** Target Encoding + Temporal Dynamics (XGBoost)
* **Level 4 (SOTA Deep Learning):** WavLM Embeddings (Neural Architecture)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/audio-quality-predictor.git](https://github.com/your-username/audio-quality-predictor.git)
   cd audio-quality-predictor
2. **Create a virtual environment (Recommended)**
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. **Install required dependencies**
    pip install -r requirements.txt
🚀 Running the Live Demo

This project includes a live interactive Streamlit dashboard for real-time MOS estimation, spectrogram generation, and audio playback.

To launch the dashboard, run:
streamlit run app.py
📊 Reproducibility & Training

To reproduce the experimental results and retrain the models from scratch, follow these steps.

Note on Data Leakage: All training scripts utilize an internal 80/20 train_test_split with a fixed random_state=42 to ensure the models are evaluated strictly on unseen data.

Feature Extraction:
Run the pipeline to extract perceptual features and statistical functionals from the raw audio:
    python pipeline.py
Model Training & Evaluation:
Execute the analysis scripts to train the regressors, calculate R-squared metrics, and generate Feature Importance graphs:
    python eval_level1.py
    python level2_analysis.py
    python informed_ml_hybrid.py
Dataset Details

This project utilizes the BVCC (Baseline Voice Conversion Challenge) dataset, provided via the VoiceMOS Challenge 2022.

Due to file size limits, the raw .wav files are not included in this repository.
    To run the code: You must obtain the VoiceMOS dataset and place the wav/ directory and sets/ directory (containing train_mos_list.txt, val_mos_list.txt, etc.) inside a folder named VoiceMos/main/DATA/ at the root of this project.