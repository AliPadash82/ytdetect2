import streamlit as st
import pandas as pd
import numpy as np
import re
import html as _html
import unicodedata as _ud
import joblib
import torch
import spacy
import os
import time
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.sparse import hstack, csr_matrix
from googleapiclient.discovery import build
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="centered")

ORDER = ["negative", "neutral", "positive", "non-english"]
CLASS_COLORS = {
    "negative": "#C0392B",
    "neutral": "#77B2B7",
    "positive": "#27AE60",
    "non-english": "#AAB7B8",
}

LOCAL_JOBLIB_PATH = "sentiment_pipeline_v1.joblib"
ROBERTA_LATEST = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ROBERTA_LEGACY = "cardiffnlp/twitter-roberta-base-sentiment"

LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect, LangDetectException, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    st.warning("langdetect not found. Install: pip install langdetect")

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    st.warning("Transformers not available.")
    device = None

@st.cache_resource
def load_spacy():
    try:
        return spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load('en_core_web_sm', disable=['parser', 'ner'])

nlp = load_spacy()
analyzer = SentimentIntensityAnalyzer()

def safe_detect(text):
    if not LANGDETECT_AVAILABLE: return 'en'
    clean = re.sub(r'[\U00010000-\U0010FFFF]', '', str(text))
    if len(clean) < 3: return 'unknown'
    try:
        return detect(clean)
    except LangDetectException:
        return 'unknown'

def extract_vid(url):
    m = re.search(r"(?:v=|youtu\.be/|/embed/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

@st.cache_resource
def load_youtube(api_key): 
    return build("youtube", "v3", developerKey=api_key)

def fetch_comments(client, vid, n_max):
    comments, token = [], None
    bar = st.progress(0, text="Fetching comments...")
    
    while len(comments) < n_max:
        response = None
        success = False
        
        for attempt in range(3):
            try:
                response = client.commentThreads().list(
                    part="snippet", videoId=vid, maxResults=100, 
                    pageToken=token, textFormat="plainText"
                ).execute()
                success = True
                break
            except Exception as e:
                if attempt == 2:
                    st.error(f"Failed after 3 attempts. Error: {e}")
                time.sleep(1)
        
        if not success:
            break
            
        for it in response.get("items", []):
            comments.append(it["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        
        token = response.get("nextPageToken")
        bar.progress(min(len(comments)/n_max, 1.0), text=f"Fetched {len(comments)} comments...")
        
        if not token or len(comments) >= n_max: 
            break
            
    bar.empty()

    # preprocess: removing URLs, Usernames and Converting HTML elements
    return comments[:n_max]

class LocalMLModel:
    def __init__(self, filepath):
        self.loaded = False
        if not os.path.exists(filepath):
            st.error(f"CRITICAL: Model file {filepath} not found.")
            return
        try:
            data = joblib.load(filepath)
            self.vectorizer = data['vectorizer']
            self.scaler = data['vader_scaler']
            self.lr = data['lr_model']
            self.svm = data['svm_model']
            self.loaded = True
        except Exception as e:
            st.error(f"Error loading joblib: {e}")

    def preprocess_batch(self, texts):
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r"[^a-z0-9\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        # clean texts for SpaCy and TF-IDF
        texts = [re.sub(r"(https?://\S+|www\.\S+|@\S+)", "", _html.unescape(str(t))) for t in texts]
        cleaned_texts = [clean_text(t) for t in texts]

        # lemmatize the cleaned text
        docs = list(nlp.pipe(cleaned_texts, batch_size=200))
        lemm = [" ".join([t.lemma_ for t in d if not t.is_space]) for d in docs]

        # VADER on uncleaned text (for better accuracy)
        vader = [analyzer.polarity_scores(str(t)) for t in texts]
        vader_feats = np.array([[v['compound'], v['neg'], v['neu'], v['pos']] for v in vader])
        
        return lemm, vader_feats

    def predict(self, texts, model_type="LR"):
        if not self.loaded or not texts: return pd.DataFrame(columns=["Comment", "Sentiment", "Confidence"])
        lemm, vader = self.preprocess_batch(texts)
        X = hstack([self.vectorizer.transform(lemm), csr_matrix(self.scaler.transform(vader))])
        
        if model_type == "LR":
            probs = self.lr.predict_proba(X)
            confs = np.max(probs, axis=1)
            preds = self.lr.predict(X)
        else:
            decs = self.svm.decision_function(X)            
            probs = 1 / (1 + np.exp(-decs))
            confs = np.max(probs, axis=1)
            preds = self.svm.predict(X)
            
        return pd.DataFrame({
            "Comment": texts, 
            "Sentiment": np.array(preds).flatten(), 
            "Confidence": np.array(confs).flatten()
        })

@st.cache_resource
def load_local_models(): return LocalMLModel(LOCAL_JOBLIB_PATH)
ml_engine = load_local_models()

@st.cache_resource
def load_roberta(model_name):
    if not TRANSFORMERS_AVAILABLE: return None, None
    local_path = f"./roberta-{'latest' if 'latest' in model_name else 'legacy'}"
    if not os.path.isdir(local_path):
        st.info(f"Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
    tok = AutoTokenizer.from_pretrained(local_path)
    mod = AutoModelForSequenceClassification.from_pretrained(local_path)
    mod.to(device).eval()
    return tok, mod

def run_roberta(texts, model_name):
    if not TRANSFORMERS_AVAILABLE: return pd.DataFrame(columns=["Comment", "Sentiment", "Confidence"])
    tok, mod = load_roberta(model_name)
    
    preds, confs = [], []
    is_legacy = "latest" not in model_name
    legacy_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    prog = st.progress(0, "Running RoBERTa...")
    total = len(texts)
    texts = [re.sub(r"(https?://\S+|www\.\S+|@\S+)", "", _html.unescape(str(t))) for t in texts]

    for i, text in enumerate(texts):
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad(): out = mod(**inp)
        probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]
        idx = np.argmax(probs)
        lbl = legacy_map[idx] if is_legacy else mod.config.id2label[idx]
        if 'neg' in lbl: lbl = 'negative'
        elif 'neu' in lbl: lbl = 'neutral'
        elif 'pos' in lbl: lbl = 'positive'
        preds.append(lbl)
        confs.append(probs[idx])
        prog.progress((i + 1) / total)
        
    prog.empty()
    return pd.DataFrame({"Comment": texts, "Sentiment": preds, "Confidence": confs})

def run_hybrid(texts, threshold, roberta_version):
    df = ml_engine.predict(texts, "LR")
    if df.empty: return df
    df["Responsible Model"] = "LR-VADER"
    
    low_conf = df[df["Confidence"] < threshold].index
    if len(low_conf) > 0:
        sub_txt = df.loc[low_conf, "Comment"].tolist()
        st.info(f"Hybrid: Sending {len(sub_txt)} comments to RoBERTa...")
        rob_df = run_roberta(sub_txt, roberta_version)
        if not rob_df.empty:
            df.loc[low_conf, "Sentiment"] = rob_df["Sentiment"].values
            df.loc[low_conf, "Confidence"] = rob_df["Confidence"].values
            df.loc[low_conf, "Responsible Model"] = "Twitter-RoBERTa"
    return df

st.title("YouTube Sentiment Analyzer")

st.subheader("Data Input")
mode = st.radio("Choose Input Method", ["YouTube API", "Upload CSV", "Text Input"], horizontal=True)

if mode == "YouTube API":
    with st.expander("API Settings", expanded=True):
        api_key = st.text_input("API Key", type="password")
        url = st.text_input("Video URL")
        nmax = st.number_input("Max Comments", 50, 50000, 100, 50)
        
        if st.button("Fetch Comments", use_container_width=True):
            if api_key and url:
                vid = extract_vid(url)
                if vid:
                    yt = load_youtube(api_key)
                    st.session_state["comments"] = fetch_comments(yt, vid, nmax)
                    st.success(f"Fetched {len(st.session_state['comments'])} comments")
                else: st.error("Invalid URL")
            else: st.error("Missing API Key or URL")

elif mode == "Upload CSV":
    with st.expander("CSV Settings", expanded=True):
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            if "Comment" in df.columns:
                n = st.slider("Select rows to analyze", 10, len(df), min(200, len(df)))
                st.session_state["comments"] = df["Comment"].dropna().astype(str).tolist()[:n]
                st.success(f"Loaded {n} comments")
            else: st.error("CSV must contain a 'Comment' column")

elif mode == "Text Input":
    with st.expander("Write a Comment", expanded=True):
        txt_in = st.text_area("Enter comment here", height=100)
        st.session_state["comments"] = [txt_in]

st.divider()

if "comments" in st.session_state and len(st.session_state["comments"]) > 0:
    st.subheader("Configuration")
    
    model_options = {
        "All Models": None,
        "Hybrid (LR-VADER -> RoBERTa)": run_hybrid,
        "LR-VADER": lambda txts, **k: ml_engine.predict(txts, "LR"),
        "LinearSVM-VADER": lambda txts, **k: ml_engine.predict(txts, "SVM"),
        "Twitter-RoBERTa": run_roberta
    }
    
    choice = st.selectbox("Select Pipeline", list(model_options.keys()))
    
    sel_rob = ROBERTA_LEGACY
    thresh = 0.9
    
    if choice in ["All Models", "Twitter-RoBERTa", "Hybrid (LR-VADER -> RoBERTa)"]:
        rob_ver = st.radio("RoBERTa Version", ["Legacy (2018)", "Latest (2021+)"], horizontal=True)
        sel_rob = ROBERTA_LATEST if "Latest" in rob_ver else ROBERTA_LEGACY
        
    if choice in ["All Models", "Hybrid (LR-VADER -> RoBERTa)"]:
        thresh = st.slider("Hybrid Confidence Threshold", 0.5, 0.99, 0.85, 0.05)
        
    use_filt = st.checkbox("Filter Non-English Comments", True)
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        raw_comments = st.session_state["comments"]
        
        final_indices = range(len(raw_comments))
        if use_filt and LANGDETECT_AVAILABLE:
            with st.spinner("Detecting Language..."):
                langs = [safe_detect(c) for c in raw_comments]
                final_indices = [i for i, l in enumerate(langs) if l == 'en']
        
        valid_comments = [raw_comments[i] for i in final_indices]
        
        to_run = [choice] if choice != "All Models" else [k for k in model_options.keys() if k != "All Models"]
        
        results = {}
        
        for m_name in to_run:
            with st.spinner(f"Running {m_name}..."):
                full_df = pd.DataFrame({"Comment": raw_comments, "Sentiment": "non-english", "Confidence": 0.0})
                if m_name == "Hybrid (LR-VADER -> RoBERTa)": full_df["Responsible Model"] = "N/A"
                
                if valid_comments:
                    func = model_options[m_name]
                    if m_name == "Hybrid (LR-VADER -> RoBERTa)":
                        res = func(valid_comments, threshold=thresh, roberta_version=sel_rob)
                    elif m_name == "Twitter-RoBERTa":
                        res = func(valid_comments, model_name=sel_rob)
                    else:
                        res = func(valid_comments)
                        
                    if not res.empty:
                        full_df.loc[final_indices, "Sentiment"] = res["Sentiment"].values
                        full_df.loc[final_indices, "Confidence"] = res["Confidence"].values
                        if "Responsible Model" in res.columns:
                            full_df.loc[final_indices, "Responsible Model"] = res["Responsible Model"].values
                
                results[m_name] = full_df
        
        st.session_state["results"] = results
else:
    st.info("Please load data above to proceed.")

if "results" in st.session_state:
    st.divider()
    st.subheader("Results")
    results = st.session_state["results"]
    
    tabs = st.tabs(results.keys())
    
    for m_name, tab in zip(results.keys(), tabs):
        with tab:
            df = results[m_name]
            counts = df["Sentiment"].value_counts()
            tot = len(df)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Pos", counts.get("positive", 0), f"{counts.get('positive',0)/tot:.1%}")
            m2.metric("Neu", counts.get("neutral", 0), f"{counts.get('neutral',0)/tot:.1%}")
            m3.metric("Neg", counts.get("negative", 0), f"{counts.get('negative',0)/tot:.1%}")
            
            c_left, c_mid, c_right = st.columns([1, 2, 1])
            with c_mid:
                st.write("**Distribution**")
                fig, ax = plt.subplots(figsize=(5, 5))
                lbls = [l for l in ORDER if l in counts.index]
                ax.pie([counts[l] for l in lbls], labels=lbls, colors=[CLASS_COLORS[l] for l in lbls], autopct="%1.1f%%", startangle=90)
                st.pyplot(fig)
            
            if "Responsible Model" in df.columns:
                st.write("**Hybrid Logic Distribution**")
                en_df = df[df["Sentiment"] != "non-english"]
                if not en_df.empty:
                    resp_counts = en_df["Responsible Model"].value_counts(normalize=True)
                    fig_bar, ax_bar = plt.subplots(figsize=(6, 0.8))
                    left = 0
                    cmap = {"LR-VADER": "#3498DB", "Twitter-RoBERTa": "#E67E22"} 
                    for i, (label, frac) in enumerate(resp_counts.items()):
                        col = cmap.get(label, "grey")
                        ax_bar.barh(0, frac, left=left, color=col, edgecolor='white', height=0.6)
                        if frac > 0.05:
                            ax_bar.text(left + frac/2, 0, f"{label}\n{frac:.1%}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
                        left += frac
                    ax_bar.set_xlim(0, 1)
                    ax_bar.axis('off')
                    st.pyplot(fig_bar)
            
            st.dataframe(
                df,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                use_container_width=True
            )
            
            st.download_button(f"Download CSV", df.to_csv(index=False).encode('utf-8'), f"{m_name}.csv", "text/csv", key=f"dl_{m_name}")

    if len(results) > 1 and "Twitter-RoBERTa" in results and mode != "Text Input":
        st.divider()
        st.subheader("Model Comparison")
        st.caption("Benchmark: Twitter-RoBERTa")
        
        baseline_df = results["Twitter-RoBERTa"]
        others = [k for k in results.keys() if k != "Twitter-RoBERTa"]
        
        for other in others:
            comp_df = results[other]
            mask = (baseline_df["Sentiment"] != "non-english") & (comp_df["Sentiment"] != "non-english")
            
            if mask.sum() > 0:
                y_true = baseline_df.loc[mask, "Sentiment"]
                y_pred = comp_df.loc[mask, "Sentiment"]
                agree = (y_true == y_pred).mean()
                
                with st.expander(f"Vs {other}", expanded=True):
                    st.metric("Agreement", f"{agree:.1%}")
                    
                    cm_labels = [l for l in ORDER if l != "non-english"]
                    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    disp = ConfusionMatrixDisplay(cm, display_labels=cm_labels)
                    disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
                    ax_cm.set_title(f"True=RoBERTa, Pred={other}")
                    st.pyplot(fig_cm)
            else:
                st.info(f"No common English comments to compare with {other}.")