"""
PART 10: STREAMLIT APP
Run (project root se):  streamlit run app.py
Model: TF-IDF + Logistic Regression (models/tfidf_lr.pkl) + LDA (models/lda.pkl)
"""
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
from config import MODELS, OUTPUTS, LABELS          # noqa: E402
from preprocess import preprocess, lemm              # noqa: E402

st.set_page_config(page_title="News Classifier", page_icon="📰", layout="wide")

CONF_THRESHOLD = 0.60
SAMPLES = {
    "(apna text likho)": "",
    "World example": "The UN Security Council met on Tuesday to discuss the ceasefire proposal as the foreign minister warned of rising tensions in the region.",
    "Sports example": "The striker scored twice in the second half as the home team won the league match and moved to the top of the table.",
    "Business example": "Shares of the airline rose after the company reported higher quarterly profit and raised its full-year forecast.",
    "Sci/Tech example": "Researchers released a new software update that improves battery life and adds an artificial intelligence assistant to the phone.",
    "Ambiguous (Business vs Sci/Tech)": "The tech company's stock fell after a regulator questioned its new search product.",
}


@st.cache_resource
def load_models():
    clf = joblib.load(MODELS / "tfidf_lr.pkl")
    lda_vec, lda = joblib.load(MODELS / "lda.pkl")
    vocab = lda_vec.get_feature_names_out()
    topic_names = []
    for i, comp in enumerate(lda.components_):
        words = [vocab[j] for j in comp.argsort()[-4:][::-1]]
        topic_names.append(f"Topic {i}: " + ", ".join(words))
    return clf, lda_vec, lda, topic_names


try:
    clf, lda_vec, lda, topic_names = load_models()
except FileNotFoundError as e:
    st.error(f"Model file nahi mili: {e}. Pehle Part 4 aur Part 7 chalao.")
    st.stop()


def top_contributions(clean, cls_idx, top_n=8):
    """Predicted class ke liye kaun se words ne sabse zyada push kiya (TF-IDF x LR weight)."""
    tfidf = clf.named_steps["vec"]
    lr = clf.named_steps["clf"]
    x = tfidf.transform([clean]).toarray().ravel()
    contrib = x * lr.coef_[cls_idx]
    names = tfidf.get_feature_names_out()
    idx = contrib.argsort()[-top_n:][::-1]
    return [(names[i], float(contrib[i])) for i in idx if contrib[i] > 0]


def highlight(text, important):
    """Original text mein important words ko orange mein dikhana."""
    words = {w for w, _ in important for w in w.split()}
    text = text.replace("$", r"\$")

    def repl(m):
        w = m.group()
        return f"**:orange[{w}]**" if lemm.lemmatize(w.lower()) in words else w

    return re.sub(r"[A-Za-z']+", repl, text)


st.title("📰 News Article Classification & Topic Discovery")
st.caption("AG News (World / Sports / Business / Sci-Tech) | TF-IDF + Logistic Regression | LDA topics")

tab1, tab2, tab3 = st.tabs(["Single article", "Upload CSV", "Model performance"])

# ---------------- Tab 1: single article ----------------
with tab1:
    choice = st.selectbox("Example chuno ya apna text likho", list(SAMPLES.keys()))
    text = st.text_area("News article / headline", value=SAMPLES[choice], height=160,
                        key=f"txt_{choice}")
    if st.button("Classify", type="primary"):
        clean = preprocess(text)
        if not clean.strip():
            st.warning("Text khali hai ya cleaning ke baad kuch bacha nahi. Thoda lamba English text daalo.")
        else:
            proba = clf.predict_proba([clean])[0]
            k = int(proba.argmax())
            conf = float(proba[k])

            c1, c2 = st.columns(2)
            c1.metric("Predicted category", LABELS[k])
            c2.metric("Confidence", f"{conf:.1%}")
            if conf < CONF_THRESHOLD:
                st.warning("Confidence kam hai. Ye article do categories ke beech ka ho sakta hai "
                           "(jaise Business vs Sci/Tech). Is prediction par zyada bharosa mat karo.")

            st.subheader("Class probabilities")
            st.bar_chart(pd.Series(proba, index=LABELS, name="probability"))

            st.subheader("Prediction ko kin words ne push kiya")
            contrib = top_contributions(clean, k)
            if contrib:
                st.markdown(highlight(text, contrib))
                st.bar_chart(pd.DataFrame(contrib, columns=["word", "contribution"]).set_index("word"))
            else:
                st.info("Is text mein koi strong keyword nahi mila.")

            st.subheader("LDA topics (unsupervised)")
            tp = lda.transform(lda_vec.transform([clean]))[0]
            st.bar_chart(pd.Series(tp, index=topic_names, name="topic share"))

# ---------------- Tab 2: CSV upload ----------------
with tab2:
    st.write("CSV upload karo, jisme articles ka ek text column ho.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        col = st.selectbox("Text wala column", df.columns)
        st.write(f"{len(df)} rows mili.")
        if st.button("Classify all"):
            if len(df) > 5000:
                st.info("Speed ke liye sirf pehli 5000 rows classify ho rahi hain.")
                df = df.head(5000)
            clean = df[col].astype(str).apply(preprocess)
            proba = clf.predict_proba(clean)
            df["predicted_category"] = [LABELS[i] for i in proba.argmax(axis=1)]
            df["confidence"] = proba.max(axis=1).round(3)
            df["low_confidence"] = df["confidence"] < CONF_THRESHOLD
            if "label" in df.columns and pd.api.types.is_numeric_dtype(df["label"]):
                acc = float((proba.argmax(axis=1) == df["label"].values).mean())
                st.success(f"CSV mein 'label' column mila. Is file par accuracy: {acc:.2%}")
            st.dataframe(df, use_container_width=True)
            st.subheader("Category distribution")
            st.bar_chart(df["predicted_category"].value_counts())
            st.download_button("Results CSV download karo",
                               df.to_csv(index=False).encode("utf-8"),
                               "predictions.csv", "text/csv")

# ---------------- Tab 3: model performance ----------------
with tab3:
    st.subheader("Model comparison (3000-article test set)")
    f = OUTPUTS / "final_comparison.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f).round(4), use_container_width=True)
    else:
        st.info("Pehle `python src/part11_compare.py` chalao.")
    cols = st.columns(2)
    for i, (fname, cap) in enumerate([("final_comparison.png", "Final comparison"),
                                      ("confusion_matrix_best.png", "Confusion matrix (TF-IDF + LR)"),
                                      ("transformer_confusion.png", "SBERT vs DistilBERT"),
                                      ("lda_topic_vs_label.png", "LDA topics vs actual labels"),
                                      ("sbert_tsne.png", "SBERT embeddings (t-SNE)"),
                                      ("class_distribution.png", "Dataset distribution")]):
        p = OUTPUTS / fname
        if p.exists():
            cols[i % 2].image(str(p), caption=cap)
    st.caption("Note: app mein TF-IDF + LR chalta hai (tez, CPU par). DistilBERT ke results Colab par nikaale gaye the.")