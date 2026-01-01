from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # project root directory

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"

J1_ROOT = RAW_DIR / "jigsaw_toxic_comments"
JIGSAW_1 = {
    "train": J1_ROOT / "train.csv",
    "test": J1_ROOT / "test.csv",
    "test_labels": J1_ROOT / "test_labels.csv",
    "combined": J1_ROOT / "jigsaw_toxic_full.csv"
}

J2_ROOT = RAW_DIR / "jigsaw_unintended_bias"
JIGSAW_2 = {
    "train": J2_ROOT / "train.csv",
    "combined": J2_ROOT / "jigsaw_bias_full.csv" 
}

# Other paths can be added here as needed