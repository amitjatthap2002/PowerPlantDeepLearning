"""Saare paths aur settings ek jagah. Baaki files yahin se import karti hain."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"              # Kaggle CSV (sirf backup ke liye)
PROCESSED = ROOT / "data" / "processed"  # cleaned train/test CSV
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"               # charts, tables, error CSV

for d in (RAW, PROCESSED, MODELS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SEED = 42
TRAIN_N = 12000   # subset size (laptop pe speed ke liye)
TEST_N = 3000
