"""
PART 1: DATA LAYER
Kaam: AG News load -> subset -> duplicates/missing hatana -> EDA charts -> CSV save
Run (project root se): python src/part1_data.py
Output: data/processed/train.csv, test.csv, sample_records.csv, outputs/class_distribution.png

Dataset link:
  HuggingFace : https://huggingface.co/datasets/fancyzhx/ag_news
  Kaggle(backup): https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
  (Kaggle se train.csv aur test.csv download karke data/raw/ mein rakho)
"""
import pandas as pd
import matplotlib.pyplot as plt
from config import RAW, PROCESSED, OUTPUTS, LABELS, SEED, TRAIN_N, TEST_N


def from_kaggle(df):
    """Kaggle CSV (Class Index, Title, Description) ko same format mein badalna."""
    return pd.DataFrame({"text": df["Title"].astype(str) + ". " + df["Description"].astype(str),
                         "label": df["Class Index"] - 1})


try:
    from datasets import load_dataset 
    ds = load_dataset("ag_news")
    train_full, test_full = ds["train"].to_pandas(), ds["test"].to_pandas()
    print("Dataset HuggingFace se load hua")
except Exception as e:
    print("HuggingFace fail:", e)
    print("Backup: data/raw/ se Kaggle CSV padh rahe hain")
    train_full = from_kaggle(pd.read_csv(RAW / "train.csv"))
    test_full = from_kaggle(pd.read_csv(RAW / "test.csv"))

print("Full dataset ->", len(train_full), "train,", len(test_full), "test")

# Subset + cleaning (duplicates / missing values)
train = train_full.sample(TRAIN_N, random_state=SEED)
test = test_full.sample(TEST_N, random_state=SEED)
before = len(train)
train = train.drop_duplicates("text").dropna().reset_index(drop=True)
test = test.drop_duplicates("text").dropna().reset_index(drop=True)
print(f"Train subset: {before} -> {len(train)} | Test subset: {len(test)}")

# EDA
train["label_name"] = train["label"].map(dict(enumerate(LABELS)))
print("\nClass distribution:\n", train["label_name"].value_counts())
train["words"] = train["text"].str.split().str.len()
print("\nAverage words per article:", round(train["words"].mean(), 1))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
train["label_name"].value_counts().plot(kind="bar", ax=ax[0], color="steelblue")
ax[0].set_title("Class Distribution")
train["words"].plot(kind="hist", bins=40, ax=ax[1], color="seagreen")
ax[1].set_title("Article Length (words)")
plt.tight_layout()
plt.savefig(OUTPUTS / "class_distribution.png", dpi=150)

train[["text", "label"]].to_csv(PROCESSED / "train.csv", index=False)
test[["text", "label"]].to_csv(PROCESSED / "test.csv", index=False)
train[["text", "label", "label_name"]].head(5).to_csv(PROCESSED / "sample_records.csv", index=False)
print("\nSample records:\n", train[["text", "label_name"]].head(3).to_string())
print("\nPART 1 DONE")
