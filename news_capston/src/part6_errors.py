"""
PART 6: ERROR ANALYSIS + EXPLAINABILITY
Run: python src/part6_errors.py
Output: outputs/misclassified_samples.csv, outputs/top_words_per_class.txt
"""
import joblib
import pandas as pd
from config import PROCESSED, MODELS, OUTPUTS, LABELS

test = pd.read_csv(PROCESSED / "test_clean.csv")
model = joblib.load(MODELS / "best_classical.pkl")

test["pred"] = model.predict(test["clean"])
err = test[test["label"] != test["pred"]].copy()
err["actual_name"] = err["label"].map(dict(enumerate(LABELS)))
err["pred_name"] = err["pred"].map(dict(enumerate(LABELS)))
print(f"Total errors: {len(err)} / {len(test)}")

print("\nSabse zyada confusion kahan hai:")
print(err.groupby(["actual_name", "pred_name"]).size().sort_values(ascending=False).head(6))

err[["text", "actual_name", "pred_name"]].head(30).to_csv(OUTPUTS / "misclassified_samples.csv", index=False)
print("\n10 galat examples (report mein reason likhna):")
for _, r in err.head(10).iterrows():
    print(f"- [{r['actual_name']} -> {r['pred_name']}] {r['text'][:130]}")

lr = joblib.load(MODELS / "tfidf_lr.pkl")
vocab = lr.named_steps["vec"].get_feature_names_out()
coefs = lr.named_steps["clf"].coef_
with open(OUTPUTS / "top_words_per_class.txt", "w") as f:
    for i, lab in enumerate(LABELS):
        top = [vocab[j] for j in coefs[i].argsort()[-15:][::-1]]
        f.write(f"{lab}: {', '.join(top)}\n")
print("\nTop words per class:\n" + open(OUTPUTS / "top_words_per_class.txt").read())
print("PART 6 DONE")
