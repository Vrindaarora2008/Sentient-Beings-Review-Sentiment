
Phase 2: UML and Package Design
1. Package Structure
The project follows a layered structure with clear separation of concerns:

data → loading, validating, de-duplicating and splitting the IMDB dataset (DatasetLoader, DatasetValidator, DataSplitter)
preprocessing → configurable text cleaning and feature representation (CleaningConfig, TextCleaner, FeatureExtractor)
models → classifiers and hyperparameter tuning (SentimentClassifier, LogisticRegressionModel, NaiveBayesModel, LinearSVMModel, NonLinearModel, ModelTuner)
experiments → ablation studies and learning curve (AblationRunner, LearningCurveGenerator)
evaluation → final metrics, coefficient interpretation and error analysis (Evaluator, CoefficientInterpreter, ErrorAnalyzer)
utils → saving results (ResultsWriter)
notebooks → entry points that run each phase (01-eda.ipynb and later notebooks)
results → generated outputs such as metrics.csv and plots
2. Core Classes
DatasetLoader
Reads IMDB Dataset.csv from data/raw/ and checks that the review and sentiment columns exist. Shows setup instructions if the file is missing (UC1, UC7).

DatasetValidator
Reports class balance, review lengths by class, missing labels, empty reviews and exact duplicates.

DataSplitter
Creates the stratified 40,000 train / 10,000 test split and keeps identical review text on one side. The test set stays locked until Phase D (UC6).

CleaningConfig
Holds the on/off switches for each preprocessing step: HTML stripping, lowercasing, punctuation removal, stop-word removal, stemming/lemmatisation, and counts vs TF-IDF.

TextCleaner
Applies the steps chosen in CleaningConfig to review text.

FeatureExtractor
Wraps CountVectorizer or TfidfVectorizer. It is fitted on training data only and keeps the matrix sparse.

SentimentClassifier
Common interface for all models: fit, predict, predict_scores.

LogisticRegressionModel, NaiveBayesModel, LinearSVMModel, NonLinearModel
The four models compared in Phase 1. They tune C, alpha, C and the non-linear model's own parameters. Logistic regression also exposes its coefficients.

ModelTuner
Runs cross-validation and hyperparameter search on the training set only.

AblationRunner
Runs the cleaning ablation: raw text, then each added step, then TF-IDF. It records cross-validated accuracy for every variant and marks each step as kept or dropped.

LearningCurveGenerator
Trains on about 1,000 reviews up to the full training portion, and plots accuracy against training size.

Evaluator
Computes accuracy, precision, recall, F1 and ROC-AUC on the held-out test set, and plots ROC curves for all models on one graph.

CoefficientInterpreter
Lists the 20 strongest positive and 20 strongest negative logistic-regression terms and plots them as a horizontal bar chart.

ErrorAnalyzer
Collects 20–30 misclassified reviews and groups them into categories such as sarcasm, mixed sentiment and sad plot summaries.

ResultsWriter
Saves results/metrics.csv, plots and split_metadata.json.

3. Design Principles
Separation of concerns between data handling, preprocessing, modelling, experiments and evaluation
No data leakage: vectorizers and models are fitted on training data only, and the 10,000-review test set is used only in Phase D
All models share one interface (SentimentClassifier), so they are trained and compared under identical conditions (UC4)
Every preprocessing step is toggled through CleaningConfig and measured, not assumed to help (UC2)
Classes map to the Phase 0 and Phase 1 use cases
Raw data and generated splits are kept out of Git
4. UML Class Diagram
classDiagram
    class DatasetLoader {
        -str path
        +load() DataFrame
        +check_columns(df) bool
    }

    class DatasetValidator {
        +class_balance(df) dict
        +review_lengths(df) Series
        +count_duplicates(df) int
        +find_invalid_rows(df) DataFrame
    }

    class DataSplitter {
        -int test_size
        -int random_state
        +split(df) tuple
        +save_split(train, test) void
    }

    class CleaningConfig {
        +bool strip_html
        +bool lowercase
        +bool remove_punctuation
        +bool remove_stopwords
        +str normalisation
        +str representation
    }

    class TextCleaner {
        -CleaningConfig config
        +clean(text) str
        +clean_all(texts) list
    }

    class FeatureExtractor {
        -CleaningConfig config
        -vectorizer
        +fit(texts) void
        +transform(texts) matrix
        +get_vocabulary() list
    }

    class SentimentClassifier {
        <<abstract>>
        +fit(X, y) void
        +predict(X) array
        +predict_scores(X) array
    }

    class LogisticRegressionModel {
        -float C
        +get_coefficients() array
    }

    class NaiveBayesModel {
        -float alpha
    }

    class LinearSVMModel {
        -float C
    }

    class NonLinearModel {
        -dict params
    }

    class ModelTuner {
        -int cv_folds
        +cross_validate(model, X, y) dict
        +tune(model, param_grid, X, y) dict
    }

    class AblationRunner {
        +run(train_df) DataFrame
        +decide_keep_or_drop(results) dict
    }

    class LearningCurveGenerator {
        +generate(model, X, y) dict
        +plot(curve) void
    }

    class Evaluator {
        +metrics(y_true, y_pred, scores) dict
        +plot_roc(models, X_test, y_test) void
    }

    class CoefficientInterpreter {
        +top_terms(coefs, vocab, n) tuple
        +plot(terms) void
    }

    class ErrorAnalyzer {
        +find_errors(texts, y_true, y_pred) DataFrame
        +group_errors(errors) dict
    }

    class ResultsWriter {
        +save_metrics(metrics, path) void
        +save_plot(fig, path) void
    }

    DatasetLoader --> DatasetValidator
    DatasetLoader --> DataSplitter
    DataSplitter --> TextCleaner
    TextCleaner --> CleaningConfig
    FeatureExtractor --> CleaningConfig
    TextCleaner --> FeatureExtractor
    FeatureExtractor --> SentimentClassifier
    SentimentClassifier <|-- LogisticRegressionModel
    SentimentClassifier <|-- NaiveBayesModel
    SentimentClassifier <|-- LinearSVMModel
    SentimentClassifier <|-- NonLinearModel
    ModelTuner --> SentimentClassifier
    AblationRunner --> TextCleaner
    AblationRunner --> FeatureExtractor
    AblationRunner --> ModelTuner
    LearningCurveGenerator --> SentimentClassifier
    Evaluator --> SentimentClassifier
    CoefficientInterpreter --> LogisticRegressionModel
    CoefficientInterpreter --> FeatureExtractor
    ErrorAnalyzer --> Evaluator
    ResultsWriter ..> Evaluator
    ResultsWriter ..> LearningCurveGenerator
