import streamlit as st
import pandas as pd
import joblib
import os
import re
import time
import io
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Mental Health AI Detection ",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0d1117; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }

    .metric-card {
        background: linear-gradient(135deg, #161b22, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    .badge-crisis {
        background: linear-gradient(135deg, #da3633, #b91c1c);
        border: 1px solid #ff7b72;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        color: white;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(218,54,51,0.4);
    }
    .badge-support {
        background: linear-gradient(135deg, #9a6700, #c67c00);
        border: 1px solid #f0b429;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        color: white;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(240,180,41,0.3);
    }
    .badge-neutral {
        background: linear-gradient(135deg, #196c2e, #238636);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        color: white;
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: 2px;
        box-shadow: 0 0 20px rgba(63,185,80,0.3);
    }

    .section-header {
        border-left: 3px solid #58a6ff;
        padding-left: 12px;
        margin: 20px 0 12px 0;
        color: #e6edf3;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    .feature-pill {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #8b949e;
        margin: 2px;
    }

    .keyword-chip {
        display: inline-block;
        background: #3d1a1a;
        border: 1px solid #da3633;
        border-radius: 16px;
        padding: 3px 10px;
        font-size: 0.82rem;
        color: #ff7b72;
        margin: 3px;
    }

    .stat-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px;
        margin: 6px 0;
    }

    hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# --- Slang Dictionary ---
SLANG = {
    'bc': 'because', 'u': 'you', 'r': 'are', 'w/': 'with',
    'idk': 'i do not know', 'rn': 'right now', 'smh': 'disappointed',
    'lol': '', 'tbh': 'to be honest', 'ngl': 'not going to lie',
    'imo': 'in my opinion', 'btw': 'by the way'
}

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'want to die', 'end my life', 'no reason to live',
    'hopeless', 'worthless', 'harm myself', 'self harm', 'give up', 'cant go on',
    'disappear', 'overdose', 'cutting', 'hurt myself'
]

# -----------------------------------------------------------------------
# Resource Loading
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models…")
def setup_nlp():
    import nltk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    vader = SentimentIntensityAnalyzer()
    le    = joblib.load('./saved_models/label_encoder.pkl')

    models = {}

    # Load all available models
    model_files = {
        "TF-IDF + LR (Augmented)": ("tfidf_augmented.pkl", "lr_augmented.pkl"),
        "TF-IDF + LR (Baseline)":  ("tfidf_baseline.pkl",  "lr_baseline.pkl"),
        "TF-IDF + RF (Augmented)": ("tfidf_augmented.pkl", "rf_augmented.pkl"),
        "TF-IDF + GB (Augmented)": ("tfidf_augmented.pkl", "gb_augmented.pkl"),
        "TF-IDF + NB (Augmented)": ("tfidf_augmented.pkl", "nb_augmented.pkl"),
        "TF-IDF + SVC (Augmented)":("tfidf_augmented.pkl", "svc_augmented.pkl"),
    }

    for name, (vec_file, clf_file) in model_files.items():
        v_path = f'./saved_models/{vec_file}'
        c_path = f'./saved_models/{clf_file}'
        if os.path.exists(v_path) and os.path.exists(c_path):
            try:
                models[name] = {
                    "encoder": joblib.load(v_path),
                    "clf":     joblib.load(c_path),
                    "mode":    "TF-IDF"
                }
            except Exception:
                pass

    # Try SBERT
    try:
        from sentence_transformers import SentenceTransformer
        sbert_enc = SentenceTransformer('./saved_models/sbert_encoder')
        sbert_clf = joblib.load('./saved_models/sbert_lr.pkl')
        models["SBERT + LR (Best)"] = {
            "encoder": sbert_enc,
            "clf":     sbert_clf,
            "mode":    "SBERT"
        }
    except Exception:
        pass

    # Fallback
    if not models:
        try:
            enc = joblib.load('./saved_models/tfidf_augmented.pkl')
            clf = joblib.load('./saved_models/lr_augmented.pkl')
            models["TF-IDF + LR (Augmented)"] = {"encoder": enc, "clf": clf, "mode": "TF-IDF"}
        except Exception:
            pass

    return models, le, vader


# -----------------------------------------------------------------------
# Text Helpers
# -----------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    for k, v in SLANG.items():
        text = re.sub(r'\b' + re.escape(k) + r'\b', v, text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_features(text: str):
    """Extract 10 linguistic features from text."""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    exclamation_count = text.count('!')
    question_count    = text.count('?')
    caps_ratio        = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    avg_word_len      = np.mean([len(w) for w in words]) if words else 0
    unique_ratio      = len(set(words)) / max(len(words), 1)
    crisis_kw_count   = sum(1 for kw in CRISIS_KEYWORDS if kw in text.lower())
    neg_words         = ['not', 'never', 'no', "don't", "can't", "won't", "shouldn't"]
    neg_count         = sum(1 for w in words if w.lower() in neg_words)
    avg_sent_len      = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    ellipsis_count    = text.count('...')
    repeat_chars      = len(re.findall(r'(.)\1{2,}', text))

    return {
        "Exclamation Marks": exclamation_count,
        "Question Marks":    question_count,
        "Caps Ratio":        round(caps_ratio, 3),
        "Avg Word Length":   round(avg_word_len, 2),
        "Vocabulary Richness": round(unique_ratio, 3),
        "Crisis Keywords":   crisis_kw_count,
        "Negation Words":    neg_count,
        "Avg Sentence Length": round(avg_sent_len, 2),
        "Ellipsis Count":    ellipsis_count,
        "Repeated Chars":    repeat_chars,
    }


def predict(text: str, model_dict: dict, le):
    encoder = model_dict["encoder"]
    clf     = model_dict["clf"]
    mode    = model_dict["mode"]
    cleaned = clean_text(text)
    if mode == "SBERT":
        features = encoder.encode([cleaned])
    else:
        features = encoder.transform([cleaned])
    pred  = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    label = le.inverse_transform([pred])[0]
    return label, proba, le.classes_


def sentiment_info(text: str, vader):
    from textblob import TextBlob
    blob   = TextBlob(text)
    scores = vader.polarity_scores(text)
    return blob, scores


def detect_crisis_keywords(text: str):
    found = [kw for kw in CRISIS_KEYWORDS if kw in text.lower()]
    return found


# -----------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------
def show_label_badge(label: str):
    css_class = f"badge-{label.lower()}"
    st.markdown(
        f'<div class="{css_class}">{label.upper()}</div>',
        unsafe_allow_html=True,
    )


def run_analysis(text: str, model_dict: dict, le, vader):
    start  = time.time()
    label, proba, classes = predict(text, model_dict, le)
    blob, v = sentiment_info(text, vader)
    feats   = extract_features(text)
    kws     = detect_crisis_keywords(text)
    latency = time.time() - start

    st.markdown('<p class="section-header">🔍 Classification Result</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        show_label_badge(label)
        st.caption("Primary Classification")

    with c2:
        conf = max(proba) * 100
        st.metric("Model Confidence", f"{conf:.1f}%")
        st.progress(conf / 100)
        st.caption(f"Engine: {model_dict['mode']}")

    with c3:
        st.metric("Latency", f"{latency*1000:.1f} ms")
        compound = v['compound']
        sentiment_label = "Positive 😊" if compound > 0.05 else ("Negative 😔" if compound < -0.05 else "Neutral 😐")
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Words:** {len(text.split())}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">📊 Class Probabilities</p>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability", ascending=False)
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}), use_container_width=True, hide_index=True)

        st.markdown('<p class="section-header">💬 Sentiment Breakdown</p>', unsafe_allow_html=True)
        sent_df = pd.DataFrame({
            "Metric": ["Compound", "Positive", "Negative", "Neutral"],
            "Score":  [v['compound'], v['pos'], v['neg'], v['neu']]
        })
        st.dataframe(sent_df.style.format({"Score": "{:.3f}"}), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown('<p class="section-header">🧮 Linguistic Features (10)</p>', unsafe_allow_html=True)
        feat_df = pd.DataFrame(list(feats.items()), columns=["Feature", "Value"])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Crisis keywords
    if kws:
        st.markdown('<p class="section-header">🚨 Detected Crisis Keywords</p>', unsafe_allow_html=True)
        chips = "".join([f'<span class="keyword-chip">⚠️ {kw}</span>' for kw in kws])
        st.markdown(chips, unsafe_allow_html=True)

    if label.lower() == "crisis":
        st.error("⚠️ **CRISIS DETECTED** — Immediate human review is recommended.")


# -----------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align:center;color:#e6edf3'>🧠 Linguistic Sentinel</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#8b949e'>Mental Health Crisis Detection System</p>", unsafe_allow_html=True)
        st.markdown("---")
        user = st.text_input("Access Key", placeholder="Enter username")
        pw   = st.text_input("Security Code", type="password", placeholder="Enter password")
        if st.button("🔐 Initialize System", use_container_width=True):
            if user == "admin" and pw == "1234":
                st.session_state["auth"] = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin / 1234")
        st.caption("Demo credentials: admin / 1234")


# -----------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------
def main_app():
    models, le, vader = setup_nlp()

    if not models:
        st.error("❌ No models found in ./saved_models/. Please upload your model files.")
        return

    st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=60)
    st.sidebar.title("Sentinel Menu")

    # Model selector
    st.sidebar.markdown("### 🤖 Model Selection")
    selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
    selected_model = models[selected_model_name]
    st.sidebar.caption(f"Engine: **{selected_model['mode']}**")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "Navigation",
        ["🏠 Core Dashboard", "📂 Batch Processing", "📋 Audit Logs"],
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 Terminate Session"):
        st.session_state["auth"] = False
        st.rerun()

    # ---- 1. Core Dashboard ----
    if menu == "🏠 Core Dashboard":
        st.title("🧠 Linguistic Sentinel")
        st.markdown("Analyze text for mental health crisis signals in real time.")
        st.markdown("---")

        input_text = st.text_area(
            "Forensic Text Input:",
            height=180,
            placeholder="Paste or type any user-generated content here…",
        )

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            scan_btn = st.button("🚀 Execute Forensic Scan", use_container_width=True, type="primary")
        with col_btn2:
            compare_btn = st.button("⚖️ Compare All Models", use_container_width=True)

        if scan_btn:
            if input_text.strip():
                run_analysis(input_text.strip(), selected_model, le, vader)
            else:
                st.warning("⚠️ Please enter some text before scanning.")

        if compare_btn:
            if input_text.strip():
                st.markdown("---")
                st.markdown('<p class="section-header">⚖️ Model Comparison</p>', unsafe_allow_html=True)
                comparison_rows = []
                for mname, mdict in models.items():
                    try:
                        t0 = time.time()
                        lbl, proba, classes = predict(input_text.strip(), mdict, le)
                        lat = (time.time() - t0) * 1000
                        comparison_rows.append({
                            "Model":      mname,
                            "Prediction": lbl,
                            "Confidence": f"{max(proba)*100:.1f}%",
                            "Latency (ms)": f"{lat:.1f}",
                        })
                    except Exception as e:
                        comparison_rows.append({
                            "Model": mname, "Prediction": "Error",
                            "Confidence": "-", "Latency (ms)": "-"
                        })
                cmp_df = pd.DataFrame(comparison_rows)
                st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Enter text first to compare models.")

    # ---- 2. Batch Processing ----
    elif menu == "📂 Batch Processing":
        st.title("📂 Batch Data Processing Hub")
        st.markdown("Upload a CSV or XLSX file with a `text` column for bulk analysis.")
        st.markdown("---")

        uploaded = st.file_uploader("Select Data Source", type=["csv", "xlsx"])
        if uploaded:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.write("**Preview (first 5 rows):**")
            st.dataframe(df.head(), use_container_width=True)

            task = st.selectbox(
                "Select Operation",
                [
                    "Batch Label Prediction",
                    "Text Cleaning",
                    "Exploratory Data Analysis",
                    "Crisis Keyword Detection",
                ],
            )

            # --- EDA sub-options ---
            chart_type = None
            eda_col = None
            if task == "Exploratory Data Analysis":
                eda_col = st.selectbox("Select column for analysis", df.columns.tolist())
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Histogram", "Pie Chart", "Box Plot (text length)", "Heatmap (correlations)"]
                )

            if st.button("▶️ Run Operation", type="primary"):

                # --- Batch Label Prediction ---
                if task == "Batch Label Prediction":
                    if "text" not in df.columns:
                        st.error("Column 'text' not found in the uploaded file.")
                    else:
                        with st.spinner("Running batch predictions…"):
                            texts   = df["text"].fillna("").tolist()
                            cleaned = [clean_text(t) for t in texts]
                            enc     = selected_model["encoder"]
                            clf     = selected_model["clf"]
                            mode    = selected_model["mode"]
                            features = enc.encode(cleaned, show_progress_bar=False) if mode == "SBERT" else enc.transform(cleaned)
                            preds   = clf.predict(features)
                            df["Predicted_Label"] = le.inverse_transform(preds)
                        st.success("✅ Batch prediction complete!")
                        st.dataframe(df[["text", "Predicted_Label"]].head(20), use_container_width=True)
                        st.markdown('<p class="section-header">📊 Label Distribution</p>', unsafe_allow_html=True)
                        st.bar_chart(df["Predicted_Label"].value_counts())

                # --- Text Cleaning ---
                elif task == "Text Cleaning":
                    if "text" not in df.columns:
                        st.error("Column 'text' not found.")
                    else:
                        df["cleaned_text"] = df["text"].apply(clean_text)
                        st.success("✅ Text cleaning complete!")
                        st.dataframe(df[["text", "cleaned_text"]].head(10), use_container_width=True)

                # --- EDA ---
                elif task == "Exploratory Data Analysis":
                    st.subheader("Statistical Summary")
                    st.write(df.describe(include="all"))

                    st.markdown(f'<p class="section-header">📊 {chart_type} — {eda_col}</p>', unsafe_allow_html=True)
                    col_data = df[eda_col].dropna()

                    if chart_type == "Bar Chart":
                        st.bar_chart(col_data.value_counts())

                    elif chart_type == "Histogram":
                        if pd.api.types.is_numeric_dtype(col_data):
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(facecolor='#161b22')
                            ax.hist(col_data, bins=30, color='#58a6ff', edgecolor='#30363d')
                            ax.set_facecolor('#0d1117')
                            ax.tick_params(colors='#e6edf3')
                            ax.spines[:].set_color('#30363d')
                            st.pyplot(fig)
                        else:
                            st.warning("Histogram requires a numeric column.")

                    elif chart_type == "Pie Chart":
                        import matplotlib.pyplot as plt
                        counts = col_data.value_counts().head(10)
                        fig, ax = plt.subplots(facecolor='#161b22')
                        ax.pie(counts.values, labels=counts.index,
                               autopct='%1.1f%%', startangle=140,
                               colors=['#58a6ff','#3fb950','#f0b429','#da3633','#a371f7',
                                       '#39d353','#ffa657','#ff7b72','#79c0ff','#d2a8ff'])
                        st.pyplot(fig)

                    elif chart_type == "Box Plot (text length)":
                        import matplotlib.pyplot as plt
                        df["_text_len"] = df[eda_col].astype(str).str.len()
                        fig, ax = plt.subplots(facecolor='#161b22')
                        ax.boxplot(df["_text_len"], patch_artist=True,
                                   boxprops=dict(facecolor='#58a6ff', color='#30363d'),
                                   medianprops=dict(color='#f0b429'))
                        ax.set_facecolor('#0d1117')
                        ax.tick_params(colors='#e6edf3')
                        ax.spines[:].set_color('#30363d')
                        ax.set_title("Text Length Distribution", color='#e6edf3')
                        st.pyplot(fig)

                    elif chart_type == "Heatmap (correlations)":
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        num_df = df.select_dtypes(include='number')
                        if num_df.shape[1] < 2:
                            st.warning("Need at least 2 numeric columns for a heatmap.")
                        else:
                            fig, ax = plt.subplots(figsize=(8, 5), facecolor='#161b22')
                            sns.heatmap(num_df.corr(), annot=True, fmt=".2f",
                                        cmap='Blues', ax=ax,
                                        linecolor='#30363d', linewidths=0.5)
                            ax.set_facecolor('#0d1117')
                            ax.tick_params(colors='#e6edf3')
                            st.pyplot(fig)

                # --- Crisis Keyword Detection ---
                elif task == "Crisis Keyword Detection":
                    if "text" not in df.columns:
                        st.error("Column 'text' not found.")
                    else:
                        df["crisis_keywords_found"] = df["text"].apply(
                            lambda x: ", ".join(detect_crisis_keywords(str(x))) if detect_crisis_keywords(str(x)) else "none"
                        )
                        df["flagged"] = df["crisis_keywords_found"] != "none"

                        flagged_df   = df[df["flagged"]]
                        unflagged_df = df[~df["flagged"]]

                        st.success(f"✅ Scan complete! {len(flagged_df)} flagged out of {len(df)} rows.")

                        col_f, col_u = st.columns(2)
                        with col_f:
                            st.markdown(f"### 🚨 Flagged ({len(flagged_df)})")
                            if not flagged_df.empty:
                                st.dataframe(flagged_df[["text", "crisis_keywords_found"]].head(20), use_container_width=True)
                        with col_u:
                            st.markdown(f"### ✅ Clean ({len(unflagged_df)})")
                            if not unflagged_df.empty:
                                st.dataframe(unflagged_df[["text"]].head(20), use_container_width=True)

                        # Keyword frequency chart
                        all_kws = []
                        for kws in df["crisis_keywords_found"]:
                            if kws != "none":
                                all_kws.extend([k.strip() for k in kws.split(",")])
                        if all_kws:
                            kw_series = pd.Series(all_kws).value_counts()
                            st.markdown('<p class="section-header">🔑 Keyword Frequency Chart</p>', unsafe_allow_html=True)
                            st.bar_chart(kw_series)

                        # Flagged vs Clean pie
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(facecolor='#161b22')
                        ax.pie(
                            [len(flagged_df), len(unflagged_df)],
                            labels=["Flagged", "Clean"],
                            autopct='%1.1f%%',
                            colors=['#da3633', '#3fb950'],
                            startangle=90,
                        )
                        ax.set_facecolor('#161b22')
                        st.markdown('<p class="section-header">📊 Flagged vs Clean</p>', unsafe_allow_html=True)
                        st.pyplot(fig)

                # --- Export always shown ---
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
                st.download_button(
                    label="⬇️ Download Processed Data (.xlsx)",
                    data=output.getvalue(),
                    file_name="sentinel_processed_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # ---- 3. Audit Logs ----
    elif menu == "📋 Audit Logs":
        st.title("📋 Security & Activity Logs")
        st.info(f"Session started at: {time.ctime()}")
        st.table(pd.DataFrame({
            "Event":     ["System Login", "NLP Resource Load", "Engine Ready"],
            "Status":    ["✅ Success", "✅ Verified", "✅ Online"],
            "Timestamp": [time.ctime()] * 3,
        }))


# -----------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login_page()
else:
    main_app()
