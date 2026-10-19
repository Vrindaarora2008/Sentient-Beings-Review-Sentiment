# P05 — What Makes a Review Negative?

## Phase A: EDA and dataset setup

This project studies sentiment classification using the IMDB Dataset of 50K Movie Reviews. This repository currently contains **Phase A only**: exploratory data analysis and an untouched stratified test holdout. Phase B will implement HTML removal, CountVectorizer and logistic regression with cross-validation.

## Dataset (download manually)

1. Visit https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Sign in to Kaggle (free account) and click **Download**.
3. Unzip the archive and locate **IMDB Dataset.csv** (approximately 66 MB).
4. Create the folder `data/raw/` and place the CSV at `data/raw/IMDB Dataset.csv`. **Do not commit this CSV to GitHub.**

The CSV contains 50,000 rows and the columns `review` and `sentiment`.

## Setup and run

Use Python 3.10+ and Jupyter Notebook or VS Code. From the project folder:

```bash
python -m pip install -r requirements.txt
jupyter notebook 01-eda.ipynb
```

Run every notebook cell in order. The notebook checks data quality, visualizes label balance and review lengths, reports exact duplicates, displays five unmodified reviews showing HTML, and creates a **40,000 training / 10,000 protected test** split with exactly 20,000/20,000 and 5,000/5,000 class balance. It keeps identical review text entirely on one side of the split.

## Files

```text
01-eda.ipynb
README.md
requirements.txt
.gitignore
data/
  raw/IMDB Dataset.csv        # download separately; ignored by Git
  splits/train.csv            # generated; ignored by Git
  splits/test.csv             # generated; ignored by Git; LOCKED until Phase D
split_metadata.json           # generated; safe to commit
```

**Test-set rule:** `test.csv` is reserved until Phase D. Do not use it for training, cleaning decisions, hyperparameter tuning, cross-validation, or baseline reporting. Phase B and C must work only with `train.csv`.

**Duplicate decision:** Keep all 50,000 original rows to meet the exact split sizes, but assign identical review texts to the same partition so there is no exact-review overlap. Record the observed duplicate counts after running the notebook.

## Phase A checklist

- [x] EDA and split code provided in `01-eda.ipynb`
- [x] README dataset instructions and Git ignore rules
- [ ] Download the CSV and execute the notebook (requires dataset)
- [ ] Check real plots/counts and write observed findings
- [ ] Commit and push project files to the team repository per Project-Handbook.md

## Source

IMDB Dataset of 50K Movie Reviews, based on Maas et al. (2011), distributed on Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

