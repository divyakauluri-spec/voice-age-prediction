VoiceAge — Age Prediction from Voice
 Age prediction from voice using CNN Deep Learning

Overview
 VoiceAge is a deep learning project that predicts a person's age from their voice. It uses a CNN architecture trained on the Mozilla Common Voice dataset with 43,737 audio samples.
Live Demo
https://voiceage-predictor.streamlit.app

Features:

-Real-time voice recording and prediction
-Voice waveform visualization
-Mel Spectrogram analysis
-Feature influence chart for explainability
-Gender-aware prediction
-Confidence score with age range

Model Architecture:

Type: CNN (Convolutional Neural Network)
Input: 185 audio features
Features: Log-Mel Spectrograms, MFCCs, Delta MFCCs, Pitch (F0), Speaking Rate
Loss Function: MAE (Mean Absolute Error)
Best MAE: 9.17 years
Training Samples: 43,737

Tech Stack

Python, PyTorch, Librosa, Streamlit, Plotly, NumPy, Scikit-learn

Installation:

bash:

-git clone https://github.com/divyakauluri-spec/voice-age-prediction.git

-cd voice-age-prediction

-pip install -r requirements.txt

-streamlit run app.py

Future Work:

-Fine-tune with VoxCeleb dataset for exact age prediction

-Add SHAP explainability

-Support audio file upload

-Mobile-friendly interface

Author:

Divya Kauluri.

GitHub: @divyakauluri-spec
LinkedIn: kauluri-divya-131888326
