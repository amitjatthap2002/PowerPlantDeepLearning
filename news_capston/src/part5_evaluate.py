"""
PART 5: EVALUATION LAYER
Kaam: models compare karna, confusion matrix, best model chunna
Run: python src/part5_evaluate.py
Output: outputs/classical_results.csv, model_comparison.png, confusion_matrix_best.png, models/best_classical.pkl
"""
import shutil
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from config import PROCESSED, MODELS, OUTPUTS, LABELS

NAMES = {"bow_nb": "BoW + Naive Bayes", "tfidf_nb": "TF-IDF + Naive Bayes",
         "tfidf_lr": "TF-IDF + Logistic Regression", "tfidf_svm": "TF-IDF + Linear SVM"}
test = pd.read_csv(PROCESSED / "test_clean.csv")

rows, preds = [], {}
for key, nice in NAMES.items():
    model = joblib.load(MODELS / f"{key}.pkl")
    pred = model.predict(test["clean"])
    preds[key] = pred
    p, r, f1, _ = precision_recall_fscore_support(test["label"], pred, average="macro")
    rows.append({"key": key, "Model": nice, "Accuracy": accuracy_score(test["label"], pred),
                 "Precision": p, "Recall": r, "F1": f1})

res = pd.DataFrame(rows).sort_values("F1", ascending=False).round(4)
res.drop(columns="key").to_csv(OUTPUTS / "classical_results.csv", index=False)
print(res.drop(columns="key").to_string(index=False))

res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", figsize=(9, 5))
plt.ylim(0.7, 1.0); plt.title("Classical Model Comparison"); plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.savefig(OUTPUTS / "model_comparison.png", dpi=150); plt.close()

best_key = res.iloc[0]["key"]
print("\nBEST MODEL:", NAMES[best_key])
print(classification_report(test["label"], preds[best_key], target_names=LABELS))
cm = confusion_matrix(test["label"], preds[best_key])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.title(f"Confusion Matrix - {NAMES[best_key]}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig(OUTPUTS / "confusion_matrix_best.png", dpi=150)

shutil.copy(MODELS / f"{best_key}.pkl", MODELS / "best_classical.pkl")
print("PART 5 DONE")
