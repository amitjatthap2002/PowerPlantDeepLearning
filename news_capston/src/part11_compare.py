"""
PART 11: FINAL MODEL COMPARISON
Kaam: classical (Part 5) + transformer (Colab) results ek table aur ek chart mein jodna
Run: python src/part11_compare.py
Pehle zaroori: outputs/classical_results.csv aur outputs/transformer_results.csv maujood ho
Output: outputs/final_comparison.csv, outputs/final_comparison.png
"""
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUTS

classical = pd.read_csv(OUTPUTS / "classical_results.csv")
transformer = pd.read_csv(OUTPUTS / "transformer_results.csv")

allr = pd.concat([classical, transformer], ignore_index=True)
allr = allr.sort_values("F1", ascending=False).reset_index(drop=True)
allr.to_csv(OUTPUTS / "final_comparison.csv", index=False)
print(allr.round(4).to_string(index=False))

best = allr.iloc[0]
print(f"\nSabse achha model: {best['Model']} (F1 = {best['F1']:.4f})")

plot = allr.set_index("Model")[["Accuracy", "F1"]].sort_values("F1")
ax = plot.plot(kind="barh", figsize=(9, 5))
lo = max(0.0, plot.values.min() - 0.03)
ax.set_xlim(lo, min(1.0, plot.values.max() + 0.02))
ax.set_title("Final Model Comparison (same 3000-article test set)")
ax.set_xlabel("Score")
plt.tight_layout()
plt.savefig(OUTPUTS / "final_comparison.png", dpi=150)
print("PART 11 DONE")