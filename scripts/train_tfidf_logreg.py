import pandas as pd
import joblib
import optuna

from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
)

from toxicity_detector.config.labels import LABELS_EN, TEXT_COL
from toxicity_detector.config.paths import (
    JIGSAW_PROCESSED,
    MODELS_DIR,
)


def print_eval_results(y_true, y_pred_prob):
    y_pred_05 = (y_pred_prob >= 0.5).astype(int)

    print(
        "Macro F1:",
        f1_score(y_true, y_pred_05, average="macro", zero_division=0),
    )

    print(
        "Micro F1:",
        f1_score(y_true, y_pred_05, average="micro", zero_division=0),
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


def build_pipeline(params=None):
    params = params or {
        "word_ngram_range": (1, 2),
        "char_ngram_range": (3, 5),
        "word_min_df": 2,
        "char_min_df": 2,
        "penalty": "l1",
        "class_weight": "balanced",
        "C": 1.0,
    }

    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=params["word_ngram_range"],
                min_df=params["word_min_df"],
                sublinear_tf=True,
                max_features=50_000,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=params["char_ngram_range"],
                min_df=params["char_min_df"],
                sublinear_tf=True,
                max_features=75_000,
            ),
        ),
    ])

    return make_pipeline(
        features,
        OneVsRestClassifier(
            LogisticRegression(
                penalty=params["penalty"],
                C=params["C"],
                solver="liblinear",
                class_weight=params["class_weight"],
                max_iter=2_000,
                random_state=42,
            )
        ),
    )


def objective(trial):
    params = {
        "word_ngram_range": trial.suggest_categorical(
            "word_ngram_range",
            [(1, 1), (1, 2)],
        ),
        "char_ngram_range": trial.suggest_categorical(
            "char_ngram_range",
            [(3, 4), (3, 5)],
        ),
        "word_min_df": trial.suggest_categorical(
            "word_min_df",
            [2, 3, 5],
        ),
        "char_min_df": trial.suggest_categorical(
            "char_min_df",
            [2, 3, 5],
        ),
        "penalty": trial.suggest_categorical(
            "penalty",
            ["l1", "l2"],
        ),
        "class_weight": trial.suggest_categorical(
            "class_weight",
            [None, "balanced"],
        ),
        "C": trial.suggest_float(
            "C",
            1e-3,
            10.0,
            log=True,
        ),
    }

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    y_val_prob = pipeline.predict_proba(X_val)
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    return f1_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0,
    )


# Loading data
train_df = pd.read_csv(JIGSAW_PROCESSED["train_uk_bin"])
val_df = pd.read_csv(JIGSAW_PROCESSED["test_uk_bin"])

X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABELS_EN].astype(int)

X_val = val_df[TEXT_COL].astype(str)
y_val = val_df[LABELS_EN].astype(int)

study = optuna.create_study(
    direction="maximize",
    study_name="tfidf_logreg_multilabel",
)

study.optimize(
    objective,
    n_trials=20,
    show_progress_bar=True,
)

print("\nBest validation macro F1:", study.best_value)
print("Best hyperparameters:")
for name, value in study.best_params.items():
    print(f"{name}: {value}")

best_pipeline = build_pipeline(study.best_params)
best_pipeline.fit(X_train, y_train)

print("\nValidation evaluation results:\n")
y_val_prob = best_pipeline.predict_proba(X_val)
print_eval_results(y_val, y_val_prob)

model_path = MODELS_DIR / "tfidf_word_char_logreg_ovr.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_pipeline, model_path, compress=3)

print(f"\nSaved best model to: {model_path.resolve()}")
print(f"Model size: {model_path.stat().st_size / 1024**2:.2f} MB")