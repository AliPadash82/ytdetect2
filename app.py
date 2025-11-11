import streamlit as st
import re
import html as _html
import unicodedata as _ud
import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import joblib
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import traceback

ORDER = ["negative", "neutral", "positive", "non-english"]
CLASS_COLORS = {
    "negative": "#C0392B",
    "neutral": "#77B2B7",
    "positive": "#27AE60",
    "non-english": "#AAB7B8",
}

ROBERTA_REMOTE_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
ROBERTA_LOCAL_PATH = "./roberta-local"

LANGDETECT_AVAILABLE = False
try:
    from langdetect import detect, LangDetectException, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    st.warning("`langdetect` not found. Non-English filtering disabled. Install: `pip install langdetect`")

TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    st.warning("Transformers (PyTorch) not available. RoBERTa model disabled.")
    device = None

try:
    import emoji as _emoji
    _HAS_EMOJI = True
except Exception: _HAS_EMOJI = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    st.error("`vaderSentiment` not found. Please install: `pip install vaderSentiment`")
    st.stop()

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

for pkg in [
    'punkt',
    'punkt_tab',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng'
]:
    try:
        nltk.download(pkg, quiet=True)
    except:
        pass


lemmatizer = WordNetLemmatizer()
vader_analyzer_for_custom = SentimentIntensityAnalyzer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens])

def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_vader_features(text):
    if pd.isna(text) or text == "": return [0.0, 0.0, 0.0, 0.0]
    scores = vader_analyzer_for_custom.polarity_scores(str(text))
    return [scores['compound'], scores['pos'], scores['neg'], scores['neu']]

_URL_RE = re.compile(r'(https?://\S+|www\.\S+)')
_MENTION_RE = re.compile(r'@\S+')
_HASHTAG_RE = re.compile(r'#(\w+)')
_TIMESTAMP_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')

def base_normalize(text: str):
    if text is None: return ""
    t = _html.unescape(str(text))
    t = _ud.normalize('NFKC', t)
    t = _URL_RE.sub(' <URL> ', t)
    t = _MENTION_RE.sub(' <USER> ', t)
    t = _HASHTAG_RE.sub(r' \1 ', t)
    t = _TIMESTAMP_RE.sub(' <TS> ', t)
    if _HAS_EMOJI: t = _emoji.demojize(t, language='en')
    return re.sub(r'\s+', ' ', t).strip()

def safe_detect(text):
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)
    if len(text) < 20:
        return 'en' if re.search(r'[A-Za-z]', text) and not re.search(r'[^\x00-\x7F]', text) else 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

# --- LOADERS ---
@st.cache_resource
def load_youtube(api_key): return build("youtube", "v3", developerKey=api_key)

@st.cache_resource
def load_roberta():
    if not TRANSFORMERS_AVAILABLE: return None, None
    if not os.path.isdir(ROBERTA_LOCAL_PATH):
        st.info(f"First-time setup: Downloading RoBERTa model to '{ROBERTA_LOCAL_PATH}'...")
        with st.spinner("Downloading and saving model... Please wait."):
            tokenizer = AutoTokenizer.from_pretrained(ROBERTA_REMOTE_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_REMOTE_NAME)
            tokenizer.save_pretrained(ROBERTA_LOCAL_PATH)
            model.save_pretrained(ROBERTA_LOCAL_PATH)
        st.success("Model downloaded successfully!")
    st.info(f"Loading RoBERTa model from local path: '{ROBERTA_LOCAL_PATH}'")
    tok = AutoTokenizer.from_pretrained(ROBERTA_LOCAL_PATH)
    mod = AutoModelForSequenceClassification.from_pretrained(ROBERTA_LOCAL_PATH)
    mod.to(device).eval()
    return tok, mod

# --- GENERIC RUNNER FOR CUSTOM LR/SVM MODELS ---
def run_custom_lr_model(txts, model_filename):
    if not txts: return pd.DataFrame()
    try:
        payload = joblib.load(model_filename)

        if isinstance(payload, dict) and 'pipeline' in payload:
            pipe = payload['pipeline']
            if 'label_mapping' in payload:
                reverse_label_mapping = {v: k for k, v in payload['label_mapping'].items()}
            else:
                reverse_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        else:
            pipe = payload
            reverse_label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}

        df_predict = pd.DataFrame(txts, columns=['Comment'])
        
        with st.spinner(f"Preprocessing for {model_filename}..."):
            df_predict['Processed_Comment'] = df_predict['Comment'].apply(preprocess_text).apply(lemmatize_text)
            vader_features = [extract_vader_features(c) for c in df_predict['Comment']]
            vader_df = pd.DataFrame(vader_features, columns=['vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral'])
            df_predict = pd.concat([df_predict, vader_df], axis=1)
            
        with st.spinner(f"Running predictions with {model_filename}..."):
            numeric_preds = pipe.predict(df_predict)
            preds = [reverse_label_mapping.get(p, 'unknown') for p in numeric_preds]

            if hasattr(pipe, "decision_function"):
                decision_values = pipe.decision_function(df_predict)
                if len(decision_values.shape) > 1:
                    probs_all = softmax(decision_values, axis=1)
                else: 
                    probs_all = 1 / (1 + np.exp(-decision_values))
                confs = [round(p.max(), 4) for p in probs_all] if len(probs_all.shape) > 1 else [round(p, 4) for p in probs_all]
            elif hasattr(pipe, "predict_proba"):
                probs_all = pipe.predict_proba(df_predict)
                confs = [round(p.max(), 4) for p in probs_all]
            else:
                confs = [1.0] * len(preds)
            
        return pd.DataFrame({"Comment": txts, "Sentiment": preds, "Confidence": confs})
    except FileNotFoundError:
        st.error(f"Model file not found: {model_filename}. Ensure it's in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error running {model_filename}: {e}")
        traceback.print_exc()
        return pd.DataFrame()


# --- MODEL RUNNERS ---
def run_roberta_twitter(txts):
    if not TRANSFORMERS_AVAILABLE or not txts: return pd.DataFrame()
    try:
        tok, mod = load_roberta()
        direct_id_map = ["negative", "neutral", "positive"]
        preds, confidences = [], []
        norm_txts = [base_normalize(t) for t in txts]
        for t in norm_txts:
            inp = tok(t, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = mod(**inp)
            probs = torch.nn.functional.softmax(out.logits, dim=-1).cpu().numpy()[0]
            top_id = int(np.argmax(probs))
            final_label = direct_id_map[top_id]
            preds.append(final_label)
            confidences.append(round(float(probs[top_id]), 4))
        return pd.DataFrame({"Comment": txts, "Sentiment": preds, "Confidence": confidences})
    except Exception as e:
        st.error(f"Error running RoBERTa model: {e}")
        return pd.DataFrame()

def run_lr_vader_custom(txts):
    return run_custom_lr_model(txts, "final_lr_sentiment_model.joblib")

def run_more_data_lr_model(txts):
    return run_custom_lr_model(txts, "more_data_lr_model.joblib")

def run_final_linear_svm_model(txts):
    return run_custom_lr_model(txts, "final_linear_svm_model.joblib")

def run_linear_svm_model(txts):
    return run_custom_lr_model(txts, "linear_svm_model.joblib")

def run_hybrid_model(txts, threshold=0.8):
    if not txts:
        return pd.DataFrame()
    with st.spinner("Running LR model..."):
        lr_results = run_more_data_lr_model(txts)
    if lr_results.empty:
        st.error("Could not get results from LR model for the hybrid approach.")
        return pd.DataFrame()
    final_sentiments = [None] * len(txts)
    final_confidences = [None] * len(txts)
    responsible_models = [None] * len(txts)
    roberta_indices_map = {}
    for i in range(len(txts)):
        lr_conf = lr_results.at[i, 'Confidence']
        if lr_conf >= threshold:
            final_sentiments[i] = lr_results.at[i, 'Sentiment']
            final_confidences[i] = lr_conf
            responsible_models[i] = "LR-VADER (trained on more data)"
        else:
            roberta_indices_map[i] = txts[i]
    if roberta_indices_map:
        roberta_indices = list(roberta_indices_map.keys())
        roberta_texts = list(roberta_indices_map.values())
        with st.spinner(f"Running RoBERTa on {len(roberta_texts)} low-confidence comments..."):
            roberta_results = run_roberta_twitter(roberta_texts)
        if not roberta_results.empty:
            roberta_results.index = roberta_indices
            for original_index, row in roberta_results.iterrows():
                final_sentiments[original_index] = row['Sentiment']
                final_confidences[original_index] = row['Confidence']
                responsible_models[original_index] = "RoBERTa"
    return pd.DataFrame({
        "Comment": txts,
        "Sentiment": final_sentiments,
        "Confidence": final_confidences,
        "Responsible Model": responsible_models
    })



def extract_vid(url):
    m = re.search(r"(?:v=|youtu\.be/|/embed/)([0-9A-Za-z_-]{11})", url)
    return m.group(1) if m else None

def fetch_comments(client, vid, n_max):
    comments, token = [], None
    bar = st.progress(0, text="Fetching comments...")
    while len(comments) < n_max:
        resp = client.commentThreads().list(part="snippet", videoId=vid, maxResults=100, pageToken=token, textFormat="plainText").execute()
        for it in resp.get("items", []):
            comments.append(it["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        token = resp.get("nextPageToken")
        bar.progress(min(len(comments)/n_max, 1.0), text=f"Fetched {len(comments)} comments...")
        if not token or len(comments) >= n_max: break
    bar.empty()
    return comments[:n_max]

st.title("YouTube Sentiment Analyzer")

mode = st.radio("Input Mode:", ["YouTube API", "Upload CSV"])
if mode == "YouTube API":
    api_key = st.text_input("YouTube API Key", type="password")
    url = st.text_input("Video URL")
    nmax = st.number_input("Max comments", 50, 5000, 100, 50)
    if st.button("Fetch Comments"):
        if not api_key or not url: st.error("Missing API key or URL"); st.stop()
        vid = extract_vid(url)
        if not vid: st.error("Invalid URL"); st.stop()
        try:
            yt = load_youtube(api_key)
            st.session_state["comments"] = fetch_comments(yt, vid, nmax)
            st.success(f"Fetched {len(st.session_state['comments'])} comments")
        except Exception as e: st.error(f"API Error: {e}")
elif mode == "Upload CSV":
    file = st.file_uploader("Upload CSV (must have 'Comment' column)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "Comment" in df.columns:
            st.session_state["comments"] = df["Comment"].dropna().astype(str).tolist()
            st.success(f"Loaded {len(st.session_state['comments'])} comments")
        else: st.error("CSV must contain a 'Comment' column.")

MODELS = {
    "Hybrid (LR-VADER -> RoBERTa)": run_hybrid_model,
    "Twitter-RoBERTa (the best)": run_roberta_twitter, 
    "LR-VADER (trained on more data)": run_more_data_lr_model,
    "LinearSVM-VADER (trained on more data)": run_final_linear_svm_model,
    "LR-VADER": run_lr_vader_custom,
    "LinearSVM-VADER": run_linear_svm_model,
}

st.divider()
filter_lang = st.toggle("Filter non-English comments", value=True, disabled=not LANGDETECT_AVAILABLE, help="Labels non-English comments instead of analyzing them.")
choice = st.selectbox("Select Model", ["All Models"] + list(MODELS.keys()))

confidence_threshold = st.slider("Confidence Threshold for LR-VADER (Hybrid Version)", 0.5, 1.0, 0.9, 0.01, help="If LR confidence is above this, their prediction is used. Otherwise, RoBERTa decides.")

if st.button("Analyze Sentiment"):
    comments = st.session_state.get("comments")
    if not comments: st.error("No comments loaded."); st.stop()
    results = {}
    to_run = MODELS.keys() if choice == "All Models" else [choice]
    lang_map = None
    if filter_lang and LANGDETECT_AVAILABLE:
        with st.spinner("Detecting languages..."):
            langs = [safe_detect(c) for c in comments]
        print(langs)
        lang_map = pd.Series(langs, index=range(len(comments)))
        st.info(f"Found {(lang_map != 'en').sum()} non-English comments.")
        
    for m_name in to_run:
        final_df = pd.DataFrame(comments, columns=['Comment'])
        # Initialize columns that might be added
        final_df['Sentiment'] = 'unknown'
        final_df['Confidence'] = 0.0
        if m_name == "Hybrid (LR-VADER -> RoBERTa)":
            final_df['Responsible Model'] = 'N/A'
            
        indices_to_analyze = final_df.index
        
        if lang_map is not None:
            en_mask, non_en_mask = (lang_map == 'en'), ~(lang_map == 'en')
            final_df.loc[non_en_mask, 'Sentiment'] = 'non-english'
            final_df.loc[non_en_mask, 'Confidence'] = 1.0
            if 'Responsible Model' in final_df.columns:
                final_df.loc[non_en_mask, 'Responsible Model'] = 'Language Filter'
            indices_to_analyze = final_df[en_mask].index
            
        comments_to_analyze = final_df.loc[indices_to_analyze, 'Comment'].tolist()
        
        if comments_to_analyze:
            # Pass threshold only to the hybrid model
            if m_name == "Hybrid (LR-VADER -> RoBERTa)":
                sentiment_results_df = MODELS[m_name](comments_to_analyze, threshold=confidence_threshold)
            else:
                sentiment_results_df = MODELS[m_name](comments_to_analyze)
            
            if not sentiment_results_df.empty:
                sentiment_results_df.index = indices_to_analyze
                for col in sentiment_results_df.columns:
                     if col in final_df.columns:
                         final_df.loc[indices_to_analyze, col] = sentiment_results_df[col].values
                
        results[m_name] = final_df
    st.session_state["results"] = results

if "results" in st.session_state and st.session_state["results"]:
    results = {k: v for k, v in st.session_state["results"].items() if not v.empty}
    if not results: st.warning("Analysis produced no results."); st.stop()
    st.divider()
    st.subheader("Analysis Results")
    tabs = st.tabs(list(results.keys()))
    for m_name, tab in zip(results.keys(), tabs):
        with tab:
            df = results[m_name]
            df["Sentiment"] = df["Sentiment"].astype(str).str.lower()
            st.write("Sentiment Breakdown")
            counts = df["Sentiment"].value_counts().reindex(ORDER, fill_value=0)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[CLASS_COLORS.get(x, "#333333") for x in counts.index], startangle=90, wedgeprops=dict(width=0.4))
            ax.axis('equal')
            st.pyplot(fig)

            if m_name == "Hybrid (LR-VADER -> RoBERTa)" and "Responsible Model" in df.columns:
                st.write("Breakdown by Responsible Model")
                resp_counts = df["Responsible Model"].value_counts(normalize=True)
                fig, ax = plt.subplots(figsize=(6, 1.2))
                left = 0
                colors = plt.cm.tab10.colors
                for i, (label, frac) in enumerate(resp_counts.items()):
                    ax.barh(0, frac, left=left, color=colors[i % len(colors)], edgecolor='white', height=0.4)
                    if frac > 0.05:
                        ax.text(left + frac / 2, 0, f"{frac*100:.1f}%", ha='center', va='center', color='white', fontsize=9, fontweight='bold')
                    left += frac
                ax.set_xlim(0, 1)
                ax.axis('off')
                ax.legend(resp_counts.index, loc='upper center', ncol=len(resp_counts), bbox_to_anchor=(0.5, 0))
                st.pyplot(fig)

            st.dataframe(df.head(50), use_container_width=True, height=300)
            st.download_button(f"Download {m_name} Results", df.to_csv(index=False).encode('utf-8'), f"{m_name.replace(' ','_')}_results.csv", "text/csv")
            
    if len(results) > 1:
        st.divider()
        st.subheader("Model Comparison against RoBERTa")
        
        baseline_model_name = "Twitter-RoBERTa (the best)"

        if baseline_model_name not in results:
            st.warning(f"'{baseline_model_name}' was not run. Cannot generate comparisons.")
        else:
            other_model_names = [name for name in results.keys() if name != baseline_model_name]

            if not other_model_names:
                st.info("Only RoBERTa was run, so there are no other models to compare it against.")
            else:
                df_baseline = results[baseline_model_name]
                
                for model_to_compare_name in other_model_names:
                    st.markdown(f"---")
                    st.write(f"#### {baseline_model_name} vs. {model_to_compare_name}")
                    df_compare = results[model_to_compare_name]
                    
                    common_english_mask = (df_baseline['Sentiment'] != 'non-english') & (df_compare['Sentiment'] != 'non-english')
                    
                    if common_english_mask.sum() > 0:
                        # calculate agreement only on this common English subset
                        agreement_score = (df_baseline[common_english_mask]['Sentiment'] == df_compare[common_english_mask]['Sentiment']).mean()
                        st.metric(f"Agreement Score", f"{agreement_score:.1%}")

                        # generate confusion matrix on the same subset
                        try:
                            cm_labels = [l for l in ORDER if l != 'non-english']
                            cm = confusion_matrix(df_baseline[common_english_mask]["Sentiment"], df_compare[common_english_mask]["Sentiment"], labels=cm_labels)
                            fig_cm, ax_cm = plt.subplots()
                            disp = ConfusionMatrixDisplay(cm, display_labels=cm_labels)
                            disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
                            ax_cm.set_title(f"Confusion Matrix (English Comments)")
                            ax_cm.set_xlabel(model_to_compare_name)
                            ax_cm.set_ylabel(baseline_model_name)
                            st.pyplot(fig_cm)
                        except Exception as e:
                            st.warning(f"Could not plot confusion matrix for this pair: {e}")
                    else:
                        st.info("No common English comments were found between these two models to compare.") 