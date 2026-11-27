"""
PART 7: TOPIC MODELING (LDA)  -- Core NLP Task #2
Kaam: bina labels ke hidden topics dhoondhna, phir dekhna ki wo asli categories se kitne match karte hain.
Run: python src/part7_lda.py
Output: outputs/lda_topic_words.png, outputs/lda_topic_vs_label.png,
        outputs/lda_topics.txt, models/lda.pkl
"""
import math
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from config import PROCESSED, MODELS, OUTPUTS, LABELS, SEED

N_TOPICS = 4      # baad mein 6 ya 8 karke bhi try karo aur result compare karo
TOP_WORDS = 10

train = pd.read_csv(PROCESSED / "train_clean.csv")

# LDA ko word counts chahiye (BoW), TF-IDF nahi
vec = CountVectorizer(max_features=5000, max_df=0.5, min_df=5)
X = vec.fit_transform(train["clean"])
print("BoW matrix:", X.shape)

lda = LatentDirichletAllocation(n_components=N_TOPICS, max_iter=20,
                                learning_method="batch", random_state=SEED)
doc_topics = lda.fit_transform(X)       # har article ka topic distribution
vocab = vec.get_feature_names_out()

# Har topic ke top words
topic_words = []
with open(OUTPUTS / "lda_topics.txt", "w") as f:
    for i, comp in enumerate(lda.components_):
        words = [vocab[j] for j in comp.argsort()[-TOP_WORDS:][::-1]]
        topic_words.append((words, comp[comp.argsort()[-TOP_WORDS:][::-1]]))
        line = f"Topic {i}: {', '.join(words)}"
        print(line)
        f.write(line + "\n")

# Chart 1: topic-wise top words
rows = math.ceil(N_TOPICS / 2)
fig, axes = plt.subplots(rows, 2, figsize=(11, 3.2 * rows))
axes = axes.ravel()
for i, (words, weights) in enumerate(topic_words):
    axes[i].barh(words[::-1], weights[::-1], color="steelblue")
    axes[i].set_title(f"Topic {i}")
for k in range(N_TOPICS, len(axes)):
    axes[k].axis("off")
plt.tight_layout()
plt.savefig(OUTPUTS / "lda_topic_words.png", dpi=150)
plt.close()

# LDA topics vs asli labels (kya discovered topics categories se match karte hain?)
train["topic"] = doc_topics.argmax(axis=1)
train["label_name"] = train["label"].map(dict(enumerate(LABELS)))
counts = pd.crosstab(train["topic"], train["label_name"])
share = counts.div(counts.sum(axis=1), axis=0)
print("\nTopic vs Label (row = topic, value = us topic ke articles ka share):")
print(share.round(2))

purity = counts.max(axis=1).sum() / counts.values.sum()
print(f"\nTopic purity: {purity:.2%}  (1 topic = 1 category hota to 100%)")

plt.figure(figsize=(7, 4.5))
sns.heatmap(share, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("LDA Topics vs Actual News Categories")
plt.ylabel("LDA topic")
plt.tight_layout()
plt.savefig(OUTPUTS / "lda_topic_vs_label.png", dpi=150)

joblib.dump((vec, lda), MODELS / "lda.pkl")
print("\nPART 7 DONE")