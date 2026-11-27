"""
PART 2: PREPROCESSING LAYER
Kaam: train/test text clean karna (preprocess.py ka function use hota hai)
Run: python src/part2_preprocess.py   (pehle part1)
Output: data/processed/train_clean.csv, test_clean.csv
"""
import pandas as pd
from config import PROCESSED
from preprocess import preprocess

train = pd.read_csv(PROCESSED / "train.csv")
test = pd.read_csv(PROCESSED / "test.csv")

for i in range(3):   # Before/After (report aur PPT ke liye)
    print("BEFORE:", train["text"][i])
    print("AFTER :", preprocess(train["text"][i]))
    print()

train["clean"] = train["text"].apply(preprocess)
test["clean"] = test["text"].apply(preprocess)
train = train[train["clean"].str.len() > 0]
test = test[test["clean"].str.len() > 0]

train.to_csv(PROCESSED / "train_clean.csv", index=False)
test.to_csv(PROCESSED / "test_clean.csv", index=False)
print("PART 2 DONE | train:", len(train), "test:", len(test))
