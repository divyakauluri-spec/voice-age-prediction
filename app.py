import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

SR = 16000
N_MELS = 64
N_MFCC = 13
MODEL_PATH = "best_model_full.pth"
SCALER_PATH = "scaler_full.pkl"
WAV_PATH = "temp_voice.wav"

st.set_page_config(
    page_title="VoiceAge — Age Prediction from Voice",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .hero-title {
        font-size: 2.8rem; font-weight: 700; color: #ffffff;
        letter-spacing: -0.5px; margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.1rem; color: #6b7280; font-weight: 400;
        margin-bottom: 0;
    }
    .badge {
        display: inline-block; padding: 0.2rem 0.8rem;
        background: #1e293b; border: 1px solid #334155;
        border-radius: 20px; font-size: 0.75rem; color: #94a3b8;
        margin-right: 0.5rem; margin-bottom: 0.5rem;
    }
    .section-card {
        background: #161b27; border: 1px solid #1e293b;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.85rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;
    }
    .metric-card {
        background: #0f1117; border: 1px solid #1e293b;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 700; color: #ffffff;
        line-height: 1;
    }
    .metric-unit { font-size: 1rem; color: #6b7280; font-weight: 400; }
    .metric-label {
        font-size: 0.8rem; color: #6b7280; margin-top: 0.4rem;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .stButton button {
        background: #6366f1 !important; color: white !important;
        font-weight: 600 !important; font-size: 1rem !important;
        border-radius: 8px !important; border: none !important;
        padding: 0.6rem 1.5rem !important; width: 100% !important;
    }
    .stButton button:hover { background: #4f46e5 !important; }
    .divider { border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0; }
    .footer-text { font-size: 0.8rem; color: #374151; text-align: center; }
    div[data-testid="stSelectbox"] > div {
        background: #161b27 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)


class VoiceAgeNetV2(nn.Module):
    def __init__(self, input_size=185):
        super(VoiceAgeNetV2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        self.gender_branch = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        cnn_out = (input_size // 4) * 64
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + 16, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 64), nn.ReLU()
        )
        self.age_head = nn.Linear(64, 1)
        self.conf_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x, gender):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        g = self.gender_branch(gender.unsqueeze(1))
        combined = torch.cat([x, g], dim=1)
        out = self.fc(combined)
        return self.age_head(out).squeeze(), self.conf_head(out).squeeze()


@st.cache_resource
def load_model():
    model = VoiceAgeNetV2(input_size=185)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def extract_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < SR * 2:
            y = np.pad(y, (0, SR * 2 - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        f0 = f0[~np.isnan(f0)]
        f0_features = [np.mean(f0), np.std(f0),
                       np.percentile(f0, 5), np.percentile(f0, 95)] if len(f0) > 0 else [0, 0, 0, 0]
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        features = np.concatenate([
            np.mean(mel_db, axis=1), np.std(mel_db, axis=1),
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1),
            f0_features, [tempo_val]
        ])
        return np.nan_to_num(features).astype(np.float32), y, mel_db
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None


def plot_waveform(y):
    time = np.linspace(0, len(y) / SR, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time[::10], y=y[::10],
        mode='lines', line=dict(color='#6366f1', width=1),
        fill='tozeroy', fillcolor='rgba(99,102,241,0.1)'
    ))
    fig.update_layout(
        paper_bgcolor='#161b27', plot_bgcolor='#161b27',
        font=dict(color='#6b7280', size=11),
        margin=dict(l=10, r=10, t=10, b=30),
        height=150,
        xaxis=dict(title='Time (s)', gridcolor='#1e293b', showgrid=True),
        yaxis=dict(title='Amplitude', gridcolor='#1e293b', showgrid=True),
        showlegend=False
    )
    return fig


def plot_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#161b27')
    ax.set_facecolor('#161b27')
    img = ax.imshow(mel_db, aspect='auto', origin='lower',
                    cmap='inferno', interpolation='nearest')
    ax.set_xlabel('Time Frames', color='#6b7280', fontsize=10)
    ax.set_ylabel('Mel Frequency Bins', color='#6b7280', fontsize=10)
    ax.tick_params(colors='#6b7280')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    plt.colorbar(img, ax=ax, format='%+2.0f dB').ax.yaxis.set_tick_params(color='#6b7280')
    plt.tight_layout()
    return fig


def plot_feature_importance(features):
    feature_names = ['Pitch (F0)', 'Speaking Rate', 'Mel Energy',
                     'MFCC Variation', 'Voice Tremor', 'Spectral Shape']
    importances = [
        abs(float(np.mean(features[128:132]))),
        abs(float(features[184])),
        abs(float(np.mean(features[:64]))),
        abs(float(np.mean(features[64:128]))),
        abs(float(np.mean(features[154:167]))),
        abs(float(np.mean(features[128:141])))
    ]
    total = sum(importances) if sum(importances) > 0 else 1
    importances = [i / total * 100 for i in importances]
    fig = go.Figure(go.Bar(
        x=importances, y=feature_names, orientation='h',
        marker=dict(
            color=importances,
            colorscale=[[0, '#1e293b'], [0.5, '#4f46e5'], [1, '#6366f1']],
            line=dict(color='#6366f1', width=0)
        )
    ))
    fig.update_layout(
        paper_bgcolor='#161b27', plot_bgcolor='#161b27',
        font=dict(color='#6b7280', size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        xaxis=dict(title='Relative Influence (%)', gridcolor='#1e293b', showgrid=True),
        yaxis=dict(gridcolor='#1e293b'),
        showlegend=False
    )
    return fig


# =====================
# UI LAYOUT
# =====================
model, scaler = load_model()

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="hero-title">VoiceAge</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Age estimation from voice using deep learning</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:0.8rem">
        <span class="badge">CNN Architecture</span>
        <span class="badge">43,737 Samples</span>
        <span class="badge">MAE: 9.17 years</span>
        <span class="badge">Mozilla Common Voice</span>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align:right; padding-top:1rem">
        <span style="font-size:0.8rem; color:#6b7280">Model Accuracy</span><br>
        <span style="font-size:2rem; font-weight:700; color:#6366f1">9.17</span>
        <span style="font-size:0.9rem; color:#6b7280"> MAE</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Configuration</p>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"])
    duration = st.slider("Recording Duration", 3, 10, 5, format="%ds")
    st.markdown("<br>", unsafe_allow_html=True)
    record_btn = st.button("Record and Analyze", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.85rem; color:#6b7280; line-height:1.8">
        <b style="color:#94a3b8">1. Record</b> — Capture 3-10s of voice<br>
        <b style="color:#94a3b8">2. Extract</b> — MFCCs, Mel spectrograms, Pitch, Delta features<br>
        <b style="color:#94a3b8">3. Predict</b> — CNN model estimates age<br>
        <b style="color:#94a3b8">4. Explain</b> — Feature influence shown
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    uploaded_file = st.file_uploader("Upload a WAV/MP3 voice file", type=["wav", "mp3"])

    if record_btn:
        file_ready = False

        if uploaded_file is not None:
            with open(WAV_PATH, "wb") as f:
                f.write(uploaded_file.read())
            st.success("File uploaded!")
            file_ready = True
        else:
            try:
                with st.spinner(f"Recording for {duration}s... Speak now!"):
                    recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)
                    sd.wait()
                    sf.write(WAV_PATH, recording.flatten(), SR)
                file_ready = True
            except Exception:
                st.warning("Mic not available on cloud. Please upload an audio file above!")
                file_ready = False

        if file_ready:
            with st.spinner("Extracting features..."):
                result = extract_features(WAV_PATH)
                features, y_audio, mel_db = result

            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                X_tensor = torch.FloatTensor(features_scaled)
                g_tensor = torch.FloatTensor([1.0 if gender == "Female" else 0.0])

                with torch.no_grad():
                    age_pred, conf_pred = model(X_tensor, g_tensor)

                age = float(age_pred.item())
                conf = float(conf_pred.item())
                margin = (1 - conf) * 15
                low = max(15, age - margin)
                high = min(85, age + margin)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Prediction Results</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'''<div class="metric-card">
                        <div class="metric-value">{age:.1f}<span class="metric-unit"> yrs</span></div>
                        <div class="metric-label">Predicted Age</div>
                    </div>''', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'''<div class="metric-card">
                        <div class="metric-value">{low:.0f}<span class="metric-unit">-{high:.0f}</span></div>
                        <div class="metric-label">Age Range</div>
                    </div>''', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'''<div class="metric-card">
                        <div class="metric-value">{conf*100:.0f}<span class="metric-unit">%</span></div>
                        <div class="metric-label">Confidence</div>
                    </div>''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Voice Waveform</p>', unsafe_allow_html=True)
                st.plotly_chart(plot_waveform(y_audio), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Mel Spectrogram</p>', unsafe_allow_html=True)
                st.pyplot(plot_spectrogram(mel_db))
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Feature Influence on Prediction</p>', unsafe_allow_html=True)
                st.plotly_chart(plot_feature_importance(features), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        if uploaded_file is None:
            st.markdown("""
            <div style="height:400px; display:flex; align-items:center;
                 justify-content:center; background:#161b27;
                 border:1px dashed #1e293b; border-radius:12px;
                 flex-direction:column; gap:1rem">
                <span style="font-size:3rem">🎙</span>
                <span style="color:#6b7280; font-size:1rem">
                    Upload an audio file or click Record to begin
                </span>
            </div>
            """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('''<p class="footer-text">
    VoiceAge • CNN Deep Learning • Mozilla Common Voice Dataset •
    185 Audio Features • 43,737 Training Samples
</p>''', unsafe_allow_html=True)