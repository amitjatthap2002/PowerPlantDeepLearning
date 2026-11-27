"""
PART 3: REPRESENTATION LAYER (BoW aur TF-IDF samajhne ke liye)
Run: python src/part3_features.py
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from config import PROCESSED

train = pd.read_csv(PROCESSED / "train_clean.csv")

bow = CountVectorizer(max_features=20000)
X_bow = bow.fit_transform(train["clean"])
print("BoW shape   :", X_bow.shape, "(documents x words)")

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
X_tfidf = tfidf.fit_transform(train["clean"])
print("TF-IDF shape:", X_tfidf.shape, "(unigrams + bigrams)")

row = X_tfidf[0].toarray().ravel()
vocab = tfidf.get_feature_names_out()
top = row.argsort()[-8:][::-1]
print("\nDocument 0:", train["text"][0])
print("Top TF-IDF words:", [(vocab[i], round(row[i], 3)) for i in top])
print("\nBoW = sirf word count | TF-IDF = common words ka weight kam, rare/important words ka zyada")
print("PART 3 DONE")
