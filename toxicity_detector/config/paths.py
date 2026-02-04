from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project root directory

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Data lifecycle folders
RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"
JIGSAW_DIR = "jigsaw"
UKR_DIR = "uk_comments"

# Jigsaw Toxic Comments Dataset (raw)
JIGSAW_TOXIC_ROOT = RAW_DIR / "jigsaw_toxic_comments"
JIGSAW_TOXIC_RAW = {
    "train": JIGSAW_TOXIC_ROOT / "train.csv",
    "test": JIGSAW_TOXIC_ROOT / "test.csv",
    "test_labels": JIGSAW_TOXIC_ROOT / "test_labels.csv",
    "combined": JIGSAW_TOXIC_ROOT / "jigsaw_toxic_full.csv",
}

# Jigsaw Unintended Bias Dataset (raw)
JIGSAW_BIAS_ROOT = RAW_DIR / "jigsaw_unintended_bias"
JIGSAW_BIAS_RAW = {
    "train": JIGSAW_BIAS_ROOT / "train.csv",
    "combined": JIGSAW_BIAS_ROOT / "jigsaw_bias_full.csv",
}

# Ukrainian comments (raw)
UKR_RAW = {
    "comments": RAW_DIR / UKR_DIR / "comments.csv",  # scraped comments (mostly toxic)
    "combined": RAW_DIR / UKR_DIR / "combined_comments.csv",  # only (mostly) toxic comments
    "non_toxic": RAW_DIR / UKR_DIR / "non_toxic_comments.csv",  # only non-toxic comments from HF dataset
    "weak_labels": RAW_DIR / UKR_DIR / "weak_labels.csv"  # labels collected using weak supervision (only toxic comments)
}

# Interim processed datasets
JIGSAW_INTERIM = {
    "train": INTERIM_DIR / JIGSAW_DIR / "train.csv",
    "test": INTERIM_DIR / JIGSAW_DIR / "test.csv",
}

UKR_INTERIM = {
    "comments": INTERIM_DIR / UKR_DIR / "comments.csv",
    "labels": INTERIM_DIR / UKR_DIR / "labels.csv",
    "test": INTERIM_DIR / UKR_DIR / "test.csv",
}

# Final processed datasets
JIGSAW_PROCESSED_DIR = PROCESSED_DIR / JIGSAW_DIR
JIGSAW_PROCESSED = {
    # Original English datasets
    "train": JIGSAW_PROCESSED_DIR / "train.csv",
    "test": JIGSAW_PROCESSED_DIR / "test.csv",
    # Ukrainian translations (imbalanced - original 1:2 ratio)
    "train_uk": JIGSAW_PROCESSED_DIR / "train_uk.csv",
    "test_uk": JIGSAW_PROCESSED_DIR / "test_uk.csv",
    # Ukrainian - binary balanced training (1:1 toxic:non-toxic)
    "train_uk_bin": JIGSAW_PROCESSED_DIR / "train_uk_bin.csv",
    "test_uk_bin": JIGSAW_PROCESSED_DIR / "test_uk_bin.csv",
}

UKR_PROCESSED_DIR = PROCESSED_DIR / UKR_DIR
UKR_PROCESSED = {
    "train": UKR_PROCESSED_DIR / "train.csv",
    "test": UKR_PROCESSED_DIR / "test.csv",
}