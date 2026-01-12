from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # project root directory

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Data lifecycle folders
RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"

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

# Interim processed datasets
JIGSAW_INTERIM = {
    "train": INTERIM_DIR / "jigsaw" / "train.csv",
    "test": INTERIM_DIR / "jigsaw" / "test.csv",
}

# Final processed datasets
JIGSAW_PROCESSED = {
    "train": PROCESSED_DIR / "jigsaw" / "train.csv",
    "test": PROCESSED_DIR / "jigsaw" / "test.csv",
    "train_ua": PROCESSED_DIR / "jigsaw" / "train_ua.csv",
    "test_ua": PROCESSED_DIR / "jigsaw" / "test_ua.csv",
}

# Other paths can be added here as needed