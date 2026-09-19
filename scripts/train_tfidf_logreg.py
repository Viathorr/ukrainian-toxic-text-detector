import pandas as pd
import joblib

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score
)

from toxicity_detector.config.labels import LABELS_EN, TEXT_COL
from toxicity_detector.config.paths import (
    JIGSAW_PROCESSED, 
    UKR_PROCESSED,
    MODELS_DIR
)


def print_eval_results(y_true, y_pred_prob):
    y_pred_05 = (y_pred_prob >= 0.5).astype(int)

    print(
        "Macro F1:",
        f1_score(y_true, y_pred_05, average="macro", zero_division=0)
    )

    print(
        "Micro F1:",
        f1_score(y_true, y_pred_05, average="micro", zero_division=0)
    )

    print("\nPer-label report at threshold = 0.50:\n")

    print(
        classification_report(
            y_true,
            y_pred_05,
            target_names=LABELS_EN,
            zero_division=0,
        )
    )

# Loading data
train_df = pd.read_csv(JIGSAW_PROCESSED["train_uk_bin"])
val_df = pd.read_csv(JIGSAW_PROCESSED["test_uk_bin"])
# test_df = pd.read_csv(UKR_PROCESSED["test"])

X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABELS_EN].astype(int)

X_val = val_df[TEXT_COL].astype(str)
y_val = val_df[LABELS_EN].astype(int)

# X_test = test_df[TEXT_COL].astype(str)
# y_test = test_df[LABELS_EN].astype(int)

# Vectorization
features = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50_000,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=75_000,
        ),
    ),
])

baseline = make_pipeline(
    features,
    OneVsRestClassifier(
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2_000,
            random_state=42
        )
    )
)

# Train
baseline.fit(X_train, y_train)
print("Baseline training complete.")

# Testing baseline performance on the validation dataset
y_val_prob = baseline.predict_proba(X_val)
print(f"Predictions shape: {y_val_prob.shape}")

print("Validation evaluation results:\n")
print_eval_results(y_val, y_val_prob)

# print("\nTest evalution results:\n")
# y_test_prob = baseline.predict_proba(X_test)
# print_eval_results(y_test, y_test_prob)

# Save the model
model_path = MODELS_DIR / "tfidf_word_char_logreg_ovr.joblib"
joblib.dump(
    baseline, 
    model_path,
    compress=3
)

print(f"Saved model to: {model_path.resolve()}")
print(f"Model size: {model_path.stat().st_size / 1024**2:.2f} MB")