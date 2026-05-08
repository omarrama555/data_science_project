import streamlit as st
import pandas as pd
import joblib
import os
import re
import time
import io

# --- Page Config ---
st.set_page_config(
    page_title="Linguistic Sentinel AI",
    page_icon="🧠",
    layout="wide"
)

# --- Slang Dictionary ---
SLANG = {
    'bc': 'because', 'u': 'you', 'r': 'are', 'w/': 'with',
    'idk': 'i do not know', 'rn': 'right now', 'smh': 'disappointed',
    'lol': '', 'tbh': 'to be honest', 'ngl': 'not going to lie',
    'imo': 'in my opinion', 'btw': 'by the way'
}

# -----------------------------------------------------------------------
# Resource Loading — cached so it only runs once
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models…")
def setup_nlp():
    import nltk
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    vader = SentimentIntensityAnalyzer()
    le    = joblib.load('./saved_models/label_encoder.pkl')

    # Try SBERT first (best model), fall back to TF-IDF + LR
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('./saved_models/sbert_encoder')
        clf     = joblib.load('./saved_models/sbert_lr.pkl')
        mode    = "SBERT"
    except Exception:
        encoder = joblib.load('./saved_models/tfidf_augmented.pkl')
        clf     = joblib.load('./saved_models/lr_augmented.pkl')
        mode    = "TF-IDF"

    return encoder, clf, le, vader, mode


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


def predict(text: str, encoder, clf, le, mode: str):
    cleaned = clean_text(text)
    if mode == "SBERT":
        features = encoder.encode([cleaned])
    else:
        features = encoder.transform([cleaned])
    pred   = clf.predict(features)[0]
    proba  = clf.predict_proba(features)[0]
    label  = le.inverse_transform([pred])[0]
    return label, proba, le.classes_


def sentiment_info(text: str, vader):
    from textblob import TextBlob
    blob   = TextBlob(text)
    scores = vader.polarity_scores(text)
    return blob, scores


# -----------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------
LABEL_COLORS = {
    "crisis":  "#d9534f",
    "support": "#f0ad4e",
    "neutral": "#5cb85c",
}

def show_label_badge(label: str):
    color = LABEL_COLORS.get(label.lower(), "#6c757d")
    st.markdown(
        f"""
        <div style="background:{color};padding:20px;border-radius:12px;text-align:center;margin-bottom:8px;">
            <h2 style="color:white;margin:0;">{label.upper()}</h2>
            <p style="color:rgba(255,255,255,0.85);margin:0;font-size:0.85rem;">Primary Classification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_analysis(text: str, encoder, clf, le, vader, mode: str):
    start   = time.time()
    label, proba, classes = predict(text, encoder, clf, le, mode)
    blob, v  = sentiment_info(text, vader)
    latency  = time.time() - start

    st.markdown("### 🔍 Diagnostic Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        show_label_badge(label)

    with c2:
        conf = max(proba) * 100
        st.metric("Model Confidence", f"{conf:.1f}%")
        st.progress(conf / 100)
        st.caption(f"Engine: {mode}")

    with c3:
        st.metric("Latency", f"{latency*1000:.1f} ms")
        st.write(f"**Subjectivity:** {blob.sentiment.subjectivity:.2f}")
        st.write(f"**Words:** {len(text.split())}")

    st.markdown("---")
    fc1, fc2 = st.columns(2)

    with fc1:
        st.subheader("📊 Class Probabilities")
        prob_df = pd.DataFrame({"Class": classes, "Probability": proba}).sort_values("Probability", ascending=False)
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}), use_container_width=True, hide_index=True)

    with fc2:
        st.subheader("💬 Sentiment Breakdown")
        sent_df = pd.DataFrame({
            "Metric":  ["Compound", "Positive", "Negative", "Neutral"],
            "Score":   [v['compound'], v['pos'], v['neg'], v['neu']]
        })
        st.dataframe(sent_df.style.format({"Score": "{:.3f}"}), use_container_width=True, hide_index=True)

    # Crisis warning banner
    if label.lower() == "crisis":
        st.error("⚠️ **CRISIS DETECTED** — Immediate human review is recommended.")


# -----------------------------------------------------------------------
# Login
# -----------------------------------------------------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align:center'>🧠 Linguistic Sentinel</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:gray'>Mental Health Crisis Detection System</p>", unsafe_allow_html=True)
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
    encoder, clf, le, vader, mode = setup_nlp()

    st.sidebar.image("https://img.icons8.com/color/96/brain.png", width=60)
    st.sidebar.title("Sentinel Menu")
    st.sidebar.caption(f"Engine: **{mode}**")

    menu = st.sidebar.radio(
        "Navigation",
        ["🏠 Core Dashboard", "📂 Batch Processing", "📋 Audit Logs", "⚙️ System Specs"],
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 Terminate Session"):
        st.session_state["auth"] = False
        st.rerun()

    # ---- 1. Core Dashboard ----
    if menu == "🏠 Core Dashboard":
        st.title("🧠 Linguistic Sentinel — Real-time Intelligence")
        st.markdown("Analyze text for mental health crisis signals in real time.")
        st.markdown("---")

        input_text = st.text_area(
            "Forensic Text Input:",
            height=180,
            placeholder="Paste or type any user-generated content here…",
        )
        if st.button("🚀 Execute Forensic Scan", use_container_width=True, type="primary"):
            if input_text.strip():
                run_analysis(input_text.strip(), encoder, clf, le, vader, mode)
            else:
                st.warning("⚠️ Please enter some text before scanning.")

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
                ["Batch Label Prediction", "Text Cleaning", "Exploratory Data Analysis"],
            )

            if st.button("▶️ Run Operation", type="primary"):
                if task == "Batch Label Prediction":
                    if "text" not in df.columns:
                        st.error("Column 'text' not found in the uploaded file.")
                    else:
                        with st.spinner("Running batch predictions…"):
                            texts = df["text"].fillna("").tolist()
                            cleaned = [clean_text(t) for t in texts]
                            if mode == "SBERT":
                                features = encoder.encode(cleaned, show_progress_bar=False)
                            else:
                                features = encoder.transform(cleaned)
                            preds = clf.predict(features)
                            df["Predicted_Label"] = le.inverse_transform(preds)
                        st.success("✅ Batch prediction complete!")
                        st.dataframe(df[["text", "Predicted_Label"]].head(20), use_container_width=True)
                        st.bar_chart(df["Predicted_Label"].value_counts())

                elif task == "Text Cleaning":
                    if "text" in df.columns:
                        df["cleaned_text"] = df["text"].apply(clean_text)
                        st.success("✅ Text cleaning complete!")
                        st.dataframe(df[["text", "cleaned_text"]].head(10), use_container_width=True)
                    else:
                        st.error("Column 'text' not found.")

                elif task == "Exploratory Data Analysis":
                    st.subheader("Statistical Summary")
                    st.write(df.describe(include="all"))
                    if "text" in df.columns:
                        df["text_len"] = df["text"].astype(str).str.len()
                        st.subheader("Text Length Distribution")
                        st.bar_chart(df["text_len"].value_counts().sort_index().head(60))

                # Export button (always shown after operation)
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

    # ---- 4. System Specs ----
    elif menu == "⚙️ System Specs":
        st.title("⚙️ System Architecture")
        st.json({
            "Primary Engine":    mode,
            "Fallback Engine":   "TF-IDF + LR (augmented)",
            "Saved Models": [
                "tfidf_baseline.pkl", "tfidf_augmented.pkl",
                "lr_baseline.pkl", "lr_augmented.pkl",
                "rf_baseline.pkl", "rf_augmented.pkl",
                "gb_baseline.pkl", "gb_augmented.pkl",
                "nb_baseline.pkl", "nb_augmented.pkl",
                "svc_baseline.pkl", "svc_augmented.pkl",
                "sbert_lr.pkl", "sbert_encoder/",
                "label_encoder.pkl",
            ],
            "Sentiment Modules": ["VADER", "TextBlob"],
            "Output Classes":    list(le.classes_),
        })
        st.markdown("---")
        st.markdown(
            "**Note:** The SBERT encoder folder (`saved_models/sbert_encoder/`) "
            "must be uploaded alongside the `.pkl` files for the full SBERT engine to load."
        )


# -----------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------
if "auth" not in st.session_state:
    st.session_state["auth"] = False

if not st.session_state["auth"]:
    login_page()
else:
    main_app()
