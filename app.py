
import streamlit as st
import pandas as pd
import joblib
import os
import re
import time
import nltk
import speech_recognition as sr
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pydub import AudioSegment
import io
import base64
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SENTINEL AI",
    page_icon="☄",
    layout="wide"
)

# ---------------- LOAD BACKGROUND ----------------
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image(
    "Crisis Communication and reputational management with SENTINEL..png"
)

# ---------------- CUSTOM UI ----------------
st.markdown(
    f"""
    <style>

    /* MAIN BACKGROUND */
    .stApp {{
        background:
        linear-gradient(rgba(0,0,0,0.45),
        rgba(0,0,0,0.45)),
        url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* REMOVE STREAMLIT DEFAULT WHITE */
    .main {{
        background: transparent !important;
    }}

    header {{
        background: transparent !important;
    }}

    /* GLOBAL TEXT */
    h1, h2, h3, h4, h5, h6 {{
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }}

    p, label, span, div {{
        color: #f1f1f1 !important;
    }}

    /* LOGIN CARD */
    .login-box {{
        background: rgba(15,15,20,0.78);
        padding: 40px;
        border-radius: 25px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 0 30px rgba(255,0,70,0.25);
        margin-top: 60px;
    }}

    /* TITLE */
    .main-title {{
        text-align: center;
        font-size: 58px;
        font-weight: 900;
        color: #ff4b6e;
        margin-bottom: 10px;
        text-shadow: 0px 0px 15px rgba(255,75,110,0.45);
    }}

    .sub-title {{
        text-align: center;
        color: #d9d9d9;
        font-size: 18px;
        margin-bottom: 35px;
    }}

    /* INPUTS */
    .stTextInput input,
    .stTextArea textarea {{
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 14px !important;
        padding: 14px !important;
        font-size: 16px !important;
    }}

    .stTextInput input:focus,
    .stTextArea textarea:focus {{
        border: 1px solid #ff4b6e !important;
        box-shadow: 0 0 12px rgba(255,75,110,0.4);
    }}

    /* BUTTON */
    .stButton button {{
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(90deg, #ff1a1a, #1a1a1a);
        color: white;
        border: none;
        font-size: 17px;
        font-weight: 700;
        padding: 14px;
        transition: 0.3s;
    }}

    .stButton button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(255,0,0,0.55);
        background: linear-gradient(90deg, #ff3333, #000000);
    }}

    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: rgba(10,10,15,0.92);
        border-right: 1px solid rgba(255,255,255,0.08);
    }}

    /* METRICS */
    [data-testid="metric-container"] {{
        background: rgba(20,20,25,0.75);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 16px;
    }}

    /* TABLES */
    .stTable {{
        background: rgba(20,20,25,0.65);
        border-radius: 12px;
    }}

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {{
        background: rgba(20,20,25,0.65) !important;
        border-radius: 14px;
        border: 1px dashed rgba(255,75,110,0.4) !important;
    }}

    /* SELECT BOX */
    .stSelectbox > div > div {{
        background-color: rgba(255,255,255,0.08) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }}

    /* DATAFRAME */
    .stDataFrame {{
        background: rgba(20,20,25,0.65);
        border-radius: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------ #
#  MODEL REGISTRY
#  Each entry: display_name -> { vectorizer, model, pca (optional) }
# ------------------------------------------------------------------ #
MODEL_REGISTRY = {
    # ── Baseline (SMOTE only) ──────────────────────────────────────
    "Logistic Regression — Baseline":        {"vec": "tfidf_baseline",  "mdl": "lr_baseline",  "pca": None},
    "Random Forest — Baseline":              {"vec": "tfidf_baseline",  "mdl": "rf_baseline",  "pca": None},
    "Gradient Boosting — Baseline":          {"vec": "tfidf_baseline",  "mdl": "gb_baseline",  "pca": None},
    "Naive Bayes — Baseline":                {"vec": "tfidf_baseline",  "mdl": "nb_baseline",  "pca": None},
    "LinearSVC — Baseline":                  {"vec": "tfidf_baseline",  "mdl": "svc_baseline", "pca": None},

    # ── Augmented (Text Aug + SMOTE) ──────────────────────────────
    "Logistic Regression — Augmented":       {"vec": "tfidf_augmented", "mdl": "lr_augmented",  "pca": None},
    "Random Forest — Augmented":             {"vec": "tfidf_augmented", "mdl": "rf_augmented",  "pca": None},
    "Gradient Boosting — Augmented":         {"vec": "tfidf_augmented", "mdl": "gb_augmented",  "pca": None},
    "Naive Bayes — Augmented":               {"vec": "tfidf_augmented", "mdl": "nb_augmented",  "pca": None},
    "LinearSVC — Augmented":                 {"vec": "tfidf_augmented", "mdl": "svc_augmented", "pca": None},

    # ── PCA / TruncatedSVD (baseline vec + svd transform + pca-retrained models) ─
    "Gradient Boosting — PCA":               {"vec": "tfidf_baseline",  "mdl": "gb_pca",  "pca": "svd"},
    "LinearSVC — PCA":                       {"vec": "tfidf_baseline",  "mdl": "svc_pca", "pca": "svd"},
}

# Models that do NOT support predict_proba (no confidence bar)
NO_PROBA_MODELS = {"svc_baseline", "svc_augmented", "svc_pca"}

SAVE_DIR = "./saved_models_v2"

# ------------------------------------------------------------------ #
#  NLP SETUP  — loads shared resources once
# ------------------------------------------------------------------ #
@st.cache_resource
def setup_nlp():
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)
    le    = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.pkl'))
    vader = SentimentIntensityAnalyzer()
    return le, vader

# ------------------------------------------------------------------ #
#  LOAD A SPECIFIC MODEL BUNDLE  — cached per name
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model_bundle(display_name: str):
    cfg = MODEL_REGISTRY[display_name]
    vec   = joblib.load(os.path.join(SAVE_DIR, f"{cfg['vec']}.pkl"))
    model = joblib.load(os.path.join(SAVE_DIR, f"{cfg['mdl']}.pkl"))
    svd   = None
    if cfg['pca']:
        svd_path = os.path.join(SAVE_DIR, 'svd.pkl')
        if os.path.exists(svd_path):
            svd = joblib.load(svd_path)
            if svd.n_components != 70:
                st.error(
                    f"❌ Loaded SVD has n_components={svd.n_components}, expected 70. "
                    "Please re-save with the correct SVD from your notebook."
                )
                svd = None
        else:
            st.warning(
                "⚠️ `svd.pkl` not found in `saved_models/`. "
                "PCA variant will run **without** dimensionality reduction. "
                "To fix: add `joblib.dump(svd, './saved_models/svd.pkl')` in your notebook after fitting TruncatedSVD.",
                icon="⚠️"
            )
    return vec, model, svd, cfg['mdl']

# ------------------------------------------------------------------ #
#  TEXT HELPERS
# ------------------------------------------------------------------ #
SLANG = {
    'bc': 'because', 'u': 'you', 'r': 'are', 'w/': 'with',
    'idk': 'i do not know', 'rn': 'right now', 'smh': 'disappointed',
    'lol': '', 'tbh': 'to be honest', 'ngl': 'not going to lie',
    'imo': 'in my opinion', 'btw': 'by the way'
}

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    for k, v in SLANG.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_batch(text):
    if pd.isna(text):
        return ""
    return clean_text(text)

# ------------------------------------------------------------------ #
#  AUDIO PROCESSING
# ------------------------------------------------------------------ #
def process_audio(audio_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_file)
    audio.export("temp.wav", format="wav")
    with sr.AudioFile("temp.wav") as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except Exception as e:
            return f"Error: Speech recognition failed. {str(e)}"

# ------------------------------------------------------------------ #
#  VECTORISE HELPER  — handles baseline / augmented / PCA
# ------------------------------------------------------------------ #
def vectorize(texts: list, vec, svd):
    X = vec.transform(texts)
    if svd is not None:
        X = svd.transform(X)          # returns dense numpy array
        X = np.array(X, dtype=float)  # ensure dense float — matches training format
    return X

# ------------------------------------------------------------------ #
#  ANALYSIS CORE
# ------------------------------------------------------------------ #
def perform_analysis(text, vec, model, svd, model_key, le, vader, display_name):
    with st.spinner("Analyzing linguistic patterns..."):
        start_time = time.time()

        cleaned = clean_text(text)
        X       = vectorize([cleaned], vec, svd)

        pred  = model.predict(X)[0]
        label = le.inverse_transform([pred])[0]

        has_proba = model_key not in NO_PROBA_MODELS
        if has_proba:
            probabilities = model.predict_proba(X)[0]

        blob     = TextBlob(text)
        v_scores = vader.polarity_scores(text)
        latency  = time.time() - start_time

        st.markdown("## Diagnostic Summary")
        st.caption(f"Engine: **{display_name}**")
        c1, c2, c3 = st.columns(3)

        with c1:
            color = (
                "#ff1a1a" if label.lower() == "crisis"
                else "#ffb347" if label.lower() == "support"
                else "#4cd964"
            )
            st.markdown(
                f"""
                <div style="
                    background:{color};
                    padding:25px;
                    border-radius:18px;
                    text-align:center;
                    box-shadow:0 0 18px rgba(0,0,0,0.3);
                ">
                    <h2 style="margin:0;color:white;">{label.upper()}</h2>
                    <p style="margin:0;color:white;">Primary Classification</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            if has_proba:
                conf = max(probabilities) * 100
                st.metric("Model Confidence", f"{conf:.2f}%")
                st.progress(conf / 100)
            else:
                st.metric("Model Confidence", "N/A (SVC)")
                st.info("LinearSVC does not output probabilities.")

        with c3:
            st.metric("Latency", f"{latency:.4f}s")
            st.metric("Subjectivity", f"{blob.sentiment.subjectivity:.2f}")

        st.markdown("---")
        x1, x2 = st.columns(2)

        with x1:
            st.subheader("Emotional Pulse")
            data = pd.DataFrame({
                "Metric": ["Compound", "Positive", "Negative", "Neutral"],
                "Value": [
                    v_scores['compound'],
                    v_scores['pos'],
                    v_scores['neg'],
                    v_scores['neu']
                ]
            })
            st.table(data)

        with x2:
            st.subheader("Linguistic Structure")
            st.write(f"**Word Count:** {len(text.split())}")
            st.write(f"**Sentence Count:** {len(blob.sentences)}")
            st.write(f"**Unique Vocabulary:** {len(set(text.lower().split()))}")

            if has_proba:
                st.subheader("Class Probabilities")
                prob_df = pd.DataFrame({
                    "Class":       le.classes_,
                    "Probability": [f"{p*100:.2f}%" for p in probabilities]
                })
                st.table(prob_df)

# ------------------------------------------------------------------ #
#  LOGIN
# ------------------------------------------------------------------ #
def login():
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown(
            """
            <div class="login-box">
            <div class="main-title">SENTINEL ☄</div>
            <div class="sub-title">Crisis Communication</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        user = st.text_input("Access Key")
        pw   = st.text_input("Security Code", type="password")
        if st.button("Initialize System"):
            if user == "admin" and pw == "1234":
                st.session_state['auth'] = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# ------------------------------------------------------------------ #
#  MAIN APP
# ------------------------------------------------------------------ #
def main_app():
    le, vader = setup_nlp()

    # ── Sidebar ────────────────────────────────────────────────────
    st.sidebar.title("SENTINEL MENU")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "Core Dashboard",
            "Voice Intelligence",
            "Data Processing Hub",
            "Audit Logs",
            "System Specs"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Model Selection")

    variant = st.sidebar.selectbox(
        "Training Variant",
        ["Baseline (SMOTE only)", "Augmented (Text Aug + SMOTE)", "PCA (TruncatedSVD)"]
    )

    variant_prefix_map = {
        "Baseline (SMOTE only)":            "Baseline",
        "Augmented (Text Aug + SMOTE)":     "Augmented",
        "PCA (TruncatedSVD)":               "PCA",
    }
    suffix = variant_prefix_map[variant]

    algo_options = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Naive Bayes", "LinearSVC"]
    if suffix == "PCA":
        algo_options = ["Gradient Boosting", "LinearSVC"]
    algo = st.sidebar.selectbox("Algorithm", algo_options)

    display_name = f"{algo} — {suffix}"
    vec, model, svd, model_key = load_model_bundle(display_name)

    st.sidebar.success(f"Active: **{display_name}**")
    st.sidebar.markdown("---")

    if st.sidebar.button("Terminate Session"):
        st.session_state['auth'] = False
        st.rerun()

    # ── 1. Core Dashboard ──────────────────────────────────────────
    if menu == "Core Dashboard":
        st.title("Linguistic Sentinel AI")
        st.markdown("---")
        input_text = st.text_area(
            "Forensic Text Input",
            height=220,
            placeholder="Enter text for forensic analysis..."
        )
        if st.button("Execute Forensic Scan"):
            if input_text.strip():
                perform_analysis(
                    input_text, vec, model, svd, model_key, le, vader, display_name
                )
            else:
                st.error("Input field is empty.")

    # ── 2. Voice Intelligence ──────────────────────────────────────
    elif menu == "Voice Intelligence":
        st.title("Voice Transcription & Analysis")
        uploaded = st.file_uploader(
            "Upload Audio Buffer",
            type=["mp3", "wav", "m4a"]
        )
        if uploaded:
            st.audio(uploaded)
            if st.button("Process Audio"):
                text = process_audio(uploaded)
                if "Error" not in text:
                    st.success("Transcription Complete")
                    st.write(f"**Transcript:** {text}")
                    st.markdown("---")
                    perform_analysis(
                        text, vec, model, svd, model_key, le, vader, display_name
                    )
                else:
                    st.error(text)

    # ── 3. Data Processing Hub ─────────────────────────────────────
    elif menu == "Data Processing Hub":
        st.title("Data Management & Transformation Hub")
        uploaded_file = st.file_uploader(
            "Select Data Source (CSV or XLSX)",
            type=["csv", "xlsx"]
        )

        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Data Preview:", df.head())

            task = st.selectbox(
                "Select Operation:",
                [
                    "Structural Data Cleaning",
                    "Exploratory Data Analysis (EDA)",
                    "Batch Label Prediction"
                ]
            )

            if st.button("Run Operation"):
                if task == "Structural Data Cleaning":
                    with st.spinner("Cleaning dataset..."):
                        df = df.drop_duplicates()
                        if 'text' in df.columns:
                            df['cleaned_text'] = df['text'].apply(clean_text_batch)
                        st.success("Cleaning complete")

                elif task == "Exploratory Data Analysis (EDA)":
                    st.subheader("Statistical Summary")
                    st.write(df.describe())
                    if 'text' in df.columns:
                        df['text_len'] = df['text'].astype(str).apply(len)
                        st.bar_chart(df['text_len'].head(50))

                elif task == "Batch Label Prediction":
                    if 'text' in df.columns:
                        with st.spinner(f"Predicting with {display_name}..."):
                            texts   = df['text'].fillna("").apply(clean_text).tolist()
                            X_batch = vectorize(texts, vec, svd)
                            preds   = model.predict(X_batch)
                            df['Predicted_Label'] = le.inverse_transform(preds)

                            # confidence where available
                            if model_key not in NO_PROBA_MODELS:
                                probs = model.predict_proba(X_batch)
                                df['Confidence_%'] = (np.max(probs, axis=1) * 100).round(2)

                            st.success(f"Batch Prediction Complete — {display_name}")
                    else:
                        st.error("Column 'text' not found in file")

                st.write(df.head())

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=output.getvalue(),
                    file_name="sentinel_processed_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ── 4. Audit Logs ──────────────────────────────────────────────
    elif menu == "Audit Logs":
        st.title("Security & Activity Logs")
        st.info(f"Current Monitoring Session: {time.ctime()}")
        st.table(pd.DataFrame({
            "Event":  ["System Login", "NLP Resource Load", "Active Model"],
            "Status": ["Success",      "Verified",          display_name]
        }))

    # ── 5. System Specs ────────────────────────────────────────────
    elif menu == "System Specs":
        st.title("System Architecture")

        # Build a registry table for the UI
        registry_rows = []
        for name, cfg in MODEL_REGISTRY.items():
            registry_rows.append({
                "Model Name":    name,
                "Vectorizer":    cfg["vec"],
                "PCA":           "Yes" if cfg["pca"] else "No",
                "Active":        "✅" if name == display_name else ""
            })

        st.subheader("Full Model Registry")
        st.dataframe(pd.DataFrame(registry_rows), use_container_width=True)

        st.subheader("Current Engine")
        st.json({
            "Active Model":     display_name,
            "Vectorizer":       MODEL_REGISTRY[display_name]["vec"],
            "PCA Applied":      MODEL_REGISTRY[display_name]["pca"] is not None,
            "Auxiliary Modules": [
                "VADER Sentiment",
                "TextBlob Linguistic",
                "Google Speech API"
            ],
            "Security Protocol": "AES-256 Auth Simulation"
        })

# ------------------------------------------------------------------ #
#  RUN
# ------------------------------------------------------------------ #
if 'auth' not in st.session_state:
    st.session_state['auth'] = False

if not st.session_state['auth']:
    login()
else:
    main_app()
