"""Shared preprocessing function (Part 2 aur baad mein Streamlit app dono use karenge)."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
lemm = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = str(text).lower()                           # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # 2. URLs hatao
    text = re.sub(r"[^a-z\s]", " ", text)              # 3. punctuation/numbers/symbols hatao
    tokens = text.split()                              # 4. tokenization
    tokens = [t for t in tokens if t not in STOP and len(t) > 2]   # 5. stop-words
    return " ".join(lemm.lemmatize(t) for t in tokens)             # 6. lemmatization
