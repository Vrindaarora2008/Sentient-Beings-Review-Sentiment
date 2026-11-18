# P05 — What Makes a Review Negative?

Investigating the drivers of sentiment in the **IMDB Dataset of 50K Movie Reviews**. We build an NLP pipeline that compares **Logistic Regression, Multinomial Naïve Bayes and a Linear SVM**, and go beyond headline accuracy with ablation studies of preprocessing choices, a TF-IDF vs. Bag-of-Words comparison, and an error analysis that interprets model coefficients and corpus artefacts.

> **Team:** Sentient Beings
> **Repository:** https://github.com/Vrindaarora2008/Sentient-Beings-Review-Sentiment

---

## Table of Contents
1. [Research Questions](#research-questions)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Instructions to Reproduce](#instructions-to-reproduce)
6. [Repository Structure](#repository-structure)
7. [Limitations](#limitations)
8. [AI-Use Acknowledgement](#ai-use-acknowledgement)
9. [Source and References](#source-and-references)

---

## Research Questions
1. Which words and phrases most strongly push a review towards *negative* sentiment?
2. How much do preprocessing decisions (HTML removal, lowercasing, stop-word removal, negation handling, n-grams) change performance?
3. Does TF-IDF outperform raw Bag-of-Words, and does the answer depend on the classifier?
4. Where do the models fail, and are some errors caused by artefacts of the corpus rather than genuine sentiment?

## Dataset
- **Source:** IMDB Dataset of 50K Movie Reviews (Maas et al., 2011), distributed on Kaggle.
- **Size:** 50,000 reviews, with columns `review` and `sentiment` (`positive` / `negative`).
- **Balance:** 25,000 positive and 25,000 negative reviews.
- **Known properties:** reviews contain raw HTML (e.g. `<br />`), vary widely in length, and include exact duplicate texts.

The CSV is **not committed** to the repository (see [Instructions](#instructions-to-reproduce)).

## Methodology

### Phase A — EDA and protected split (`01-eda-checkpoint.ipynb`)
- Data-quality checks: missing values, label balance, review-length distribution, exact duplicates.
- Inspection of unmodified reviews to confirm HTML artefacts.
- A **stratified 40,000 train / 10,000 test split** (20,000/20,000 and 5,000/5,000 per class).
- **Duplicate handling:** all 50,000 rows are kept, but identical review texts are assigned to the *same* partition so no exact review appears in both train and test.
- Split details are recorded in `split_metadata.json`.
- **Test-set rule:** `test.csv` stays locked until the final evaluation phase. It is never used for training, cleaning decisions, hyperparameter tuning, cross-validation or baselines.

### Phase B — Baseline
- HTML tag removal.
- `CountVectorizer` (Bag-of-Words) features.
- Logistic Regression, evaluated with stratified k-fold cross-validation on `train.csv` only.

### Phase C — Model comparison and ablations
All experiments use only the training split (cross-validation for tuning, plus a validation split for comparison).

| Factor | Variants compared |
|---|---|
| Classifier | Logistic Regression, Multinomial Naïve Bayes, Linear SVM, Decision Tree (as a non-linear reference) |
| Representation | Bag-of-Words (`CountVectorizer`) vs. TF-IDF (`TfidfVectorizer`) |
| Preprocessing | HTML removal, lowercasing, stop-word removal, punctuation handling, negation handling |
| N-gram range | Unigrams vs. unigrams + bigrams |
| Vocabulary | `min_df` / `max_features` settings |

Ablations change **one factor at a time** against the baseline so each effect can be isolated. Hyperparameters (e.g. `C`, NB smoothing `alpha`) are tuned with cross-validated grid search on the training data.

### Phase D — Final evaluation and interpretation
- The selected configuration is trained on the full training split and evaluated **once** on the locked test set.
- **Metrics:** accuracy, precision, recall, F1 and the confusion matrix. Because classes are balanced, accuracy is meaningful, but we also report per-class metrics to expose asymmetries.
- **Coefficient analysis:** the top positive and negative features from the linear models.
- **Error analysis:** manual review of false positives and false negatives, grouped by cause (sarcasm, negation, mixed sentiment, plot summaries, very long reviews, corpus artefacts).

## Results

All figures below come from the CSV files in `results/`. Model selection used the training data only (cross-validation and a validation split); the test set was evaluated once at the end.

### Phase A: dataset observations (`results/eda_summary.csv`)
| Check | Observed value |
|---|---|
| Total reviews | 50,000 |
| Positive / negative | 25,000 / 25,000 |
| Missing values | 0 |
| Duplicate reviews | 418 |
| Unique reviews | 49,582 |
| Duplicate texts with conflicting labels | 0 |

### Phase B: baseline (`results/baseline_metrics.csv`)
Logistic Regression + `CountVectorizer` (Bag-of-Words) + HTML removal.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.8855 | 0.8792 | 0.8938 | 0.8864 | 0.9515 |

### Vectorizer and model comparison (default settings, validation data)
All configurations use the same vocabulary size (84,297 features). Sorted by F1.

| Vectorizer | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---|---|---|---|---|---|---|
| TF-IDF | Linear SVM | 0.8994 | 0.8907 | 0.9105 | 0.9005 | 0.9640 | 13.5 |
| TF-IDF | Logistic Regression | 0.8921 | 0.8832 | 0.9038 | 0.8934 | 0.9602 | 14.4 |
| Bag-of-Words | Logistic Regression | 0.8855 | 0.8792 | 0.8938 | 0.8864 | 0.9515 | 32.7 |
| Bag-of-Words | Linear SVM | 0.8688 | 0.8642 | 0.8750 | 0.8696 | 0.9380 | 24.1 |
| TF-IDF | Multinomial Naïve Bayes | 0.8641 | 0.8818 | 0.8410 | 0.8609 | 0.9382 | 13.1 |
| Bag-of-Words | Multinomial Naïve Bayes | 0.8489 | 0.8722 | 0.8175 | 0.8440 | 0.9190 | 13.6 |

### Hyperparameter tuning: cross-validation (`results/cross_validation_results.csv`)
| Model | Best CV F1 | Best parameters | Tuning time (s) |
|---|---|---|---|
| Logistic Regression | 0.8960 | `C=10`, `class_weight=None` | 127.8 |
| Linear SVM | 0.8939 | `C=1`, `class_weight='balanced'` | 136.5 |
| Multinomial Naïve Bayes | 0.8550 | `alpha=0.1`, `fit_prior=False` | 113.4 |
| Decision Tree | 0.7434 | `max_depth=20`, `min_samples_split=2`, `min_samples_leaf=10`, `class_weight='balanced'` | 371.3 |

### Tuned models on the validation set (`results/tuned_model_validation_results.csv`)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9008 | 0.8941 | 0.9093 | 0.9016 | 0.9641 |
| Linear SVM | 0.8994 | 0.8907 | 0.9105 | 0.9005 | 0.9640 |
| Multinomial Naïve Bayes | 0.8664 | 0.8719 | 0.8590 | 0.8654 | 0.9364 |
| Decision Tree | 0.7249 | 0.6963 | 0.7978 | 0.7436 | 0.7761 |

### Learning curve
| Training samples | Training F1 | Validation F1 | Training time (s) |
|---|---|---|---|
| 1,000 | 1.0000 | 0.8429 | 0.43 |
| 5,000 | 0.9972 | 0.8803 | 1.50 |
| 10,000 | 0.9949 | 0.9070 | 3.76 |
| 20,000 | 0.9892 | 0.9423 | 6.97 |
| 30,000 | 0.9868 | 0.9643 | 10.47 |
| 32,000 | 0.9853 | 0.9684 | 9.94 |

<!-- TODO: confirm which model/split produced this learning curve. Validation F1 (0.968) is higher than the 0.90 reported elsewhere. -->

### Final test-set performance (evaluated once, `results/final_test_metrics.csv`)
| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.8958 | 0.8879 | 0.9060 | 0.8969 | 0.9632 |

### Preprocessing ablation
| Variant | F1 | Δ vs. baseline |
|---|---|---|
| Baseline (HTML removal only) | 0.8864 | — |
| + Lowercasing | TBD | TBD |
| + Stop-word removal | TBD | TBD |
| + Negation handling | TBD | TBD |
| + Bigrams | TBD | TBD |

### Interpretation
- **TF-IDF beat Bag-of-Words for every classifier.** F1 gains on the validation set were about +0.031 for Linear SVM (0.8696 → 0.9005), +0.007 for Logistic Regression (0.8864 → 0.8934) and +0.017 for Naïve Bayes (0.8440 → 0.8609). TF-IDF also trained faster for Logistic Regression and Linear SVM.
- **Linear models clearly outperformed the others.** After tuning, Logistic Regression (F1 0.9016) and Linear SVM (0.9005) were nearly tied, Naïve Bayes trailed (0.8654), and the Decision Tree performed far worse (0.7436), as expected for a sparse, high-dimensional text feature space.
- **Naïve Bayes favours precision over recall** (recall 0.8590 vs. precision 0.8719), whereas the linear models favour recall.
- **Tuning gave modest gains**: Logistic Regression improved from 0.8934 (default, TF-IDF) to 0.9016 F1 on validation.
- **Generalisation:** the final test F1 (0.8969) is close to the validation F1 (0.9016), so there is no sign of overfitting to the validation data.
- **Most negative / positive features:** TBD (top coefficients from the linear model).
- **Error themes and corpus artefacts:** TBD (sarcasm, negation, mixed sentiment, and the 418 duplicate reviews).

## Instructions to Reproduce

**Requirements:** Python 3.10+, and Jupyter Notebook or VS Code.

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vrindaarora2008/Sentient-Beings-Review-Sentiment.git
   cd Sentient-Beings-Review-Sentiment
   ```
2. **(Optional) create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```
4. **Download the dataset manually**
   - Go to https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews and sign in (free account).
   - Download and unzip the archive, then locate `IMDB Dataset.csv` (about 66 MB).
   - Create `data/raw/` and place the file at `data/raw/IMDB Dataset.csv`. **Do not commit it.**
5. **Run the notebooks in order**, executing every cell top to bottom:
   ```bash
   jupyter notebook 01-eda-checkpoint.ipynb   # Phase A: EDA + creates data/splits/train.csv and test.csv
   ```
   Then run the remaining notebooks in `notebooks/` in numerical order (baseline, ablations, final evaluation).
6. **Outputs** (figures, tables, metrics) are written to `results/`.

**Reproducibility:** the split uses a fixed random seed recorded in `split_metadata.json`. Use the same seed for all stochastic steps.

## Repository Structure
```
01-eda-checkpoint.ipynb      # Phase A: EDA and protected split
notebooks/                   # Modelling, ablation and analysis notebooks
results/                     # Figures, tables and metrics
data/
  raw/IMDB Dataset.csv       # download separately; ignored by Git
  splits/train.csv           # generated; ignored by Git
  splits/test.csv            # generated; ignored by Git; LOCKED until Phase D
split_metadata.json          # generated; safe to commit
Phase 0- Use Cases and Problem Statement
Phase-1-Usecases.md
Phase-2-UML.md
requirements.txt
.gitignore
README.md
```

## Limitations
- Bag-of-Words and TF-IDF ignore word order and long-range context, so sarcasm and complex negation remain hard.
- IMDB reviews are polarised (binary labels, with neutral reviews excluded), so results may not transfer to other domains or to graded or neutral sentiment.
- Coefficients show correlation with the label, not causation, and may reflect corpus-specific artefacts (genres, franchises, reviewer style).
- Exact duplicates are separated across splits, but near-duplicates may still exist.

## AI-Use Acknowledgement
Generative AI (Anthropic's Claude) was used in this project only for **verification and debugging**: checking our code and results for mistakes, and helping diagnose errors encountered while running the notebooks. *TBD: list specifics if desired (e.g. checking the train/test split for leakage, debugging vectorizer or cross-validation errors).*

All code was run by the team, and all metrics come from our own executed notebooks. Any AI suggestions were reviewed and tested by the team, who remain responsible for the design, results and conclusions.

## Source and References
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
- IMDB Dataset of 50K Movie Reviews (Kaggle): https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
