"""
PART 4: CLASSICAL MODELING LAYER
Kaam: 4 models train karna (BoW+NB, TF-IDF+NB, TF-IDF+LR, TF-IDF+SVM)
Run: python src/part4_models.py
Output: models/*.pkl
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from config import PROCESSED, MODELS

train = pd.read_csv(PROCESSED / "train_clean.csv")


def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=30000)


experiments = {
    "bow_nb": Pipeline([("vec", CountVectorizer(max_features=20000)), ("clf", MultinomialNB())]),
    "tfidf_nb": Pipeline([("vec", tfidf()), ("clf", MultinomialNB())]),
    "tfidf_lr": Pipeline([("vec", tfidf()), ("clf", LogisticRegression(max_iter=1000))]),
    "tfidf_svm": Pipeline([("vec", tfidf()), ("clf", LinearSVC())]),
}

for name, pipe in experiments.items():
    pipe.fit(train["clean"], train["label"])
    joblib.dump(pipe, MODELS / f"{name}.pkl")
    print("Trained & saved:", name)
print("PART 4 DONE")
