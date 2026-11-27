# News Article Classification and Topic Discovery

Capstone project for **ETMMTI303 - Language Technologies and Text Intelligence** (MCA AI & ML, Semester III).

An end-to-end NLP application that classifies news articles into four categories (World, Sports, Business, Sci/Tech) and discovers hidden topics, with a Streamlit interface.

## Pipeline

Raw text -> Preprocessing -> BoW / TF-IDF / Sentence-BERT -> Classical baselines -> DistilBERT -> Evaluation -> Error analysis -> Streamlit app -> Visualization

## Dataset

- **AG News** (HuggingFace: https://huggingface.co/datasets/fancyzhx/ag_news)
- Backup source (Kaggle): https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
- Full size: 120,000 train / 7,600 test, 4 balanced classes.
- Used here: a random subset of **12,000 train / 3,000 test** articles (seed 42), chosen so that training runs on a laptop.
- Columns: `text` (title + description), `label` (0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech).

## Folder structure

```
news_capston/
├── app.py                  # Streamlit app
├── requirements.txt
├── data/raw/               # Kaggle CSV (only if HuggingFace is blocked)
├── data/processed/         # train/test CSVs (created by the scripts)
├── src/
│   ├── config.py           # paths and settings
│   ├── preprocess.py       # text cleaning function
│   ├── part1_data.py       # dataset load + EDA
│   ├── part2_preprocess.py
│   ├── part3_features.py   # BoW / TF-IDF
│   ├── part4_models.py     # NB / LR / SVM
│   ├── part5_evaluate.py   # metrics, confusion matrix
│   ├── part6_errors.py     # error analysis, top words
│   ├── part7_lda.py        # topic modeling
│   └── part11_compare.py   # final comparison table and chart
├── notebooks/part8_9_transformers_colab.ipynb   # SBERT + DistilBERT (Google Colab, GPU)
├── models/                 # saved models
├── outputs/                # charts, tables, error CSVs
└── docs/                   # report, presentation
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Run the scripts from the project root, in this order:

```bash
python src/part1_data.py
python src/part2_preprocess.py
python src/part3_features.py
python src/part4_models.py
python src/part5_evaluate.py
python src/part6_errors.py
python src/part7_lda.py
```

**Transformers (Parts 8-9):** open `notebooks/part8_9_transformers_colab.ipynb` in Google Colab with a T4 GPU, upload `data/processed/train.csv` and `test.csv`, run all cells, and copy the downloaded files (`transformer_results.csv`, `sbert_tsne.png`, `transformer_confusion.png`, `bert_misclassified.csv`) into `outputs/`.

Then:

```bash
python src/part11_compare.py
streamlit run app.py
```

## Results (3,000-article test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| DistilBERT (fine-tuned) | 0.9250 | 0.9247 |
| TF-IDF + Logistic Regression | 0.9010 | 0.9003 |
| TF-IDF + Linear SVM | 0.9007 | 0.8999 |
| TF-IDF + Naive Bayes | 0.8947 | 0.8940 |
| BoW + Naive Bayes | 0.8940 | 0.8936 |
| SBERT + Logistic Regression | 0.8883 | 0.8880 |

- The most common error is **Business <-> Sci/Tech** confusion (tech-company business news).
- LDA (4 topics) matches the true categories with about 79% purity.

## Limitations

- Results come from a single train/test split and a single random seed on a 12k training subset, so very small differences between classical models are not conclusive.
- AG News contains some label noise and HTML artifacts, and the model partly learns news-agency names (for example "afp").
- The Streamlit app uses TF-IDF + Logistic Regression for speed; DistilBERT was evaluated on Google Colab and is not loaded in the app.
- English text only.

## References

- Zhang, Zhao, LeCun (2015). Character-level Convolutional Networks for Text Classification (AG News).
- Sanh et al. (2019). DistilBERT, a distilled version of BERT.
- Reimers and Gurevych (2019). Sentence-BERT.
- Blei, Ng, Jordan (2003). Latent Dirichlet Allocation.
- scikit-learn, HuggingFace Transformers, sentence-transformers, Streamlit documentation.