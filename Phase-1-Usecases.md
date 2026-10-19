# Phase 1: Requirements and Use Cases

## 1. Actors
- Student / Data Scientist
- Evaluator / Instructor

---

## 2. Main Use Cases
- Load IMDB Review Dataset
- Explore and Validate Dataset
- Detect and Handle Duplicate Reviews
- Split Training and Test Data
- Preprocess Review Text
- Perform Cleaning Ablation
- Train Sentiment Classification Models
- Tune Model Hyperparameters
- Evaluate Models
- Interpret Logistic Regression Coefficients
- Generate Learning Curve
- Perform Error Analysis
- Save Results and Documentation

---

## 3. Detailed Use Cases

### Use Case 1: Load IMDB Review Dataset
- Actor: Student / Data Scientist
- Description: Student loads the IMDB Dataset of 50K Movie Reviews from the local CSV file.
- Precondition: `IMDB Dataset.csv` has been downloaded from Kaggle and placed in the expected local data directory.
- Main Flow:
  1. Student starts the EDA notebook
  2. System reads the CSV file
  3. System checks the `review` and `sentiment` columns
  4. System reports the number of rows and columns
  5. Dataset is made available for analysis
- Failure Case:
  - Dataset file is missing → system shows download/setup instructions
  - Required columns are missing → system reports invalid dataset format

---

### Use Case 2: Explore and Validate Dataset
- Actor: Student / Data Scientist
- Description: Student examines the class balance, review lengths, and raw review text before modelling.
- Precondition: Dataset has been loaded successfully.
- Main Flow:
  1. System counts positive and negative reviews
  2. System verifies the expected class balance
  3. System calculates review lengths
  4. System compares review-length distributions by sentiment
  5. System displays five raw reviews
  6. Student observes HTML such as `<br /><br />` in the text
- Failure Case:
  - Missing sentiment values → system reports incomplete labels
  - Empty review text → system identifies invalid rows

---

### Use Case 3: Detect and Handle Duplicate Reviews
- Actor: Student / Data Scientist
- Description: Student identifies duplicate reviews and makes a documented decision about their treatment.
- Precondition: Dataset has been loaded.
- Main Flow:
  1. System checks the `review` column for duplicates
  2. System reports the duplicate count
  3. Student decides whether duplicates should be removed
  4. Selected duplicate-handling rule is applied before the train/test split
  5. Decision is documented in the notebook and report
- Failure Case:
  - Duplicate reviews are allowed to occur across training and test sets → possible leakage is reported as a problem

---

### Use Case 4: Split Training and Test Data
- Actor: Student / Data Scientist
- Description: Student creates an untouched stratified test set of 10,000 reviews and keeps the remaining data for development.
- Precondition: Dataset validation and duplicate handling are complete.
- Main Flow:
  1. System separates features and sentiment labels
  2. System performs a stratified train/test split
  3. Exactly 10,000 reviews are assigned to the test set
  4. Remaining reviews are assigned to the training/development portion
  5. Test set is kept untouched during preprocessing experiments and tuning
- Failure Case:
  - Test data is used during vectorizer fitting, ablation, or tuning → data leakage occurs
  - Split is not stratified → class proportions may change unnecessarily

---

### Use Case 5: Build Baseline Text Classification Pipeline
- Actor: Student / Data Scientist
- Description: Student builds the minimum baseline using HTML stripping, CountVectorizer, and logistic regression.
- Precondition: Training portion has been created and the test set is isolated.
- Main Flow:
  1. System strips HTML from review text
  2. CountVectorizer converts training text into bag-of-words features
  3. Logistic regression is trained inside the pipeline
  4. Cross-validation is performed only on the training portion
  5. Mean cross-validated accuracy is reported as the baseline
- Failure Case:
  - Vectorizer is fitted on the full dataset → leakage
  - Model is evaluated repeatedly on the held-out test set → test set is no longer untouched

---

### Use Case 6: Perform Cleaning Ablation
- Actor: Student / Data Scientist
- Description: Student measures the effect of each preprocessing decision instead of assuming that cleaning improves accuracy.
- Precondition: Baseline pipeline is working.
- Main Flow:
  1. Evaluate raw text
  2. Add HTML stripping and evaluate again
  3. Add lowercasing and evaluate again
  4. Add punctuation removal and evaluate again
  5. Add stop-word removal and evaluate again
  6. Add stemming or lemmatisation and evaluate again
  7. Replace raw counts with TF-IDF and evaluate again
  8. Record cross-validated accuracy for every variant
  9. Mark whether each preprocessing step is kept or dropped
  10. Keep only preprocessing choices that improve the selected representation
- Failure Case:
  - A preprocessing step reduces accuracy → step is dropped and the reduction is explained
  - Stop-word removal deletes useful negation terms such as `not`, `no`, or `never` → sentiment information may be lost

---

### Use Case 7: Train and Tune Classification Models
- Actor: Student / Data Scientist
- Description: Student compares linear and non-linear classifiers using the selected text representation.
- Precondition: Final preprocessing representation has been selected from the ablation.
- Main Flow:
  1. Train logistic regression and tune `C` using cross-validation
  2. Train multinomial naïve Bayes and tune `alpha`
  3. Train LinearSVC and tune `C`
  4. Train one selected non-linear model
  5. Record cross-validation performance for each model
  6. Document the loss/objective associated with each model
  7. Explain why non-linear models can be a poor computational trade for high-dimensional sparse text features
- Failure Case:
  - Hyperparameters are selected using the held-out test set → invalid evaluation
  - Dense conversion of a very large sparse matrix causes excessive memory use → model/pipeline must preserve sparse features where possible

---

### Use Case 8: Evaluate Models on the Held-Out Test Set
- Actor: Student / Data Scientist
- Description: Student performs final evaluation on the untouched 10,000-review test set.
- Precondition: Preprocessing choices and model hyperparameters have already been finalized using training data only.
- Main Flow:
  1. Train each final model using the training portion
  2. Predict sentiment for the held-out test reviews
  3. Calculate accuracy
  4. Calculate precision
  5. Calculate recall
  6. Calculate F1 score
  7. Calculate ROC-AUC using appropriate decision scores/probabilities
  8. Save model metrics to `results/metrics.csv`
  9. Plot ROC curves for the models on one graph
- Failure Case:
  - A model does not expose `predict_proba` → use an appropriate decision score when supported for ROC-AUC
  - Test set has already influenced model selection → final metrics are considered biased

---

### Use Case 9: Interpret Logistic Regression
- Actor: Student / Data Scientist
- Description: Student identifies the words that most strongly drive positive and negative predictions.
- Precondition: Final logistic regression model has been fitted with an interpretable vocabulary representation.
- Main Flow:
  1. Extract the logistic-regression coefficients
  2. Match coefficients to vocabulary terms
  3. Select the 20 strongest negative coefficients
  4. Select the 20 strongest positive coefficients
  5. Plot the terms in a horizontal bar chart
  6. Identify obvious sentiment words
  7. Identify at least one corpus artefact
  8. Explain why the artefact may have received a large coefficient
- Failure Case:
  - Coefficients are interpreted without matching the correct vectorizer vocabulary → incorrect word interpretation

---

### Use Case 10: Generate Learning Curve
- Actor: Student / Data Scientist
- Description: Student measures how classification accuracy changes as more training reviews are used.
- Precondition: A final model and representation have been selected.
- Main Flow:
  1. Train/evaluate using approximately 1,000 training reviews
  2. Repeat for progressively larger training sizes
  3. Continue up to the available training portion, approximately 40,000 reviews before any duplicate-removal adjustment
  4. Plot accuracy against training-set size
  5. Identify where performance begins to flatten
  6. Explain whether collecting more data or improving the model/representation is likely to help more
- Failure Case:
  - Test data is used to repeatedly choose the learning-curve configuration → test leakage

---

### Use Case 11: Perform Error Analysis
- Actor: Student / Data Scientist
- Description: Student manually reads misclassified reviews to understand model failure patterns.
- Precondition: Final predictions on the held-out test set are available.
- Main Flow:
  1. Find incorrectly classified reviews
  2. Select 20–30 misclassified examples for manual inspection
  3. Read the review text
  4. Group errors into meaningful categories
  5. Consider categories such as sarcasm, mixed sentiment, sad plot summaries, and DVD/product-focused reviews
  6. Include an example for each reported category
  7. Explain why a bag-of-words linear classifier struggles with each category
- Failure Case:
  - Error categories are generated without reading the reviews → analysis does not satisfy the project requirement

---

### Use Case 12: Save Results and Documentation
- Actor: Student / Data Scientist
- Description: Student prepares the repository so the experiment can be reproduced from a clean checkout.
- Precondition: Analysis, modelling, and evaluation are complete.
- Main Flow:
  1. Save required plots and `results/metrics.csv`
  2. Complete the project report
  3. Update README with dataset download instructions
  4. Add `IMDB Dataset.csv` to `.gitignore`
  5. Complete `REPRODUCIBILITY.md`
  6. Verify notebooks run in the documented order
  7. Commit and push final project files
- Failure Case:
  - Large dataset CSV is committed to Git → repository rules are violated
  - Missing setup instructions prevent reproduction → documentation is incomplete

---

## 4. Acceptance Criteria

### For Dataset Preparation:
- IMDB 50K review dataset must load successfully
- Positive and negative label counts must be reported
- Review-length distributions must be examined by class
- Five raw reviews must be displayed to show the HTML issue
- Duplicate reviews must be counted
- Duplicate-handling decision must be documented
- A stratified test set of 10,000 reviews must be held out from the start
- Held-out test data must not be used during preprocessing selection or hyperparameter tuning

### For Cleaning Ablation:
- Raw text must be evaluated
- HTML stripping must be evaluated
- Lowercasing must be evaluated
- Punctuation removal must be evaluated
- Stop-word removal must be evaluated
- Stemming or lemmatisation must be evaluated
- TF-IDF must be compared with raw counts
- Every variant must have a real cross-validated accuracy
- Steps that reduce performance must be identified and dropped
- Reasons for harmful preprocessing steps must be discussed

### For Model Training:
- Logistic regression must be trained and `C` tuned by cross-validation
- Multinomial naïve Bayes must be trained and `alpha` tuned by cross-validation
- LinearSVC must be trained and `C` tuned by cross-validation
- One non-linear model must be evaluated
- Model selection must use training data only
- The loss/objective for the compared models must be discussed

### For Final Evaluation:
- Accuracy must be reported for every final model
- Precision must be reported
- Recall must be reported
- F1 score must be reported
- ROC-AUC must be reported
- ROC curves must be plotted on one axis
- Final metrics must be saved to `results/metrics.csv`

### For Interpretation:
- Top 20 positive logistic-regression terms must be identified
- Top 20 negative logistic-regression terms must be identified
- Coefficients must be shown using a horizontal bar chart
- Sentiment-related terms must be discussed
- At least one corpus artefact must be identified and explained

### For Learning Curve:
- Training sizes must begin around 1,000 reviews and extend to the full available training portion
- Accuracy must be plotted against training-set size
- The point where performance begins to flatten must be discussed
- The report must explain the implication for collecting more data versus improving the model

### For Error Analysis:
- 20–30 misclassified reviews must be manually inspected
- Misclassifications must be grouped into failure types
- Each reported failure type must include an example
- The reason for the model failure must be explained

### For Repository and Reproducibility:
- `IMDB Dataset.csv` must be excluded through `.gitignore`
- README must contain exact dataset download/setup instructions
- Report must be included
- `REPRODUCIBILITY.md` must be included
- Project must run from a clean checkout after following the documented setup steps

---

## 5. Main Workflow Focus
Dataset Download → EDA → Duplicate Handling → Stratified Split → Baseline → Cleaning Ablation → Representation Selection → Model Tuning → Final Test Evaluation → Coefficient Interpretation → Learning Curve → Error Analysis → Report and Reproducibility

---

## 6. Optional Extensions
- Add word bigrams using `ngram_range=(1, 2)` and test whether phrases such as `not good` improve negation handling
- Test character n-grams
- Sweep vocabulary size using `max_features` and plot vocabulary size versus accuracy
- Cluster TF-IDF review vectors without sentiment labels and compare clusters with true sentiment using adjusted Rand index
- Train on the first 25,000 reviews and test on the second 25,000 to investigate possible corpus ordering

---

## 7. Important Constraints
- Do not call `fit_transform` on the full corpus before the train/test split
- Do not allow duplicate reviews to leak across training and test data
- Do not use the 10,000-review test set for ablation or hyperparameter tuning
- Do not assume additional text cleaning is beneficial; measure every preprocessing step
- Do not report only accuracy; interpretation and error analysis are required
- Do not commit the approximately 66 MB dataset CSV to the repository
