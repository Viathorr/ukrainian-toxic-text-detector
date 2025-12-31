from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # project root directory

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DIR = DATA_DIR / "01_raw"
INTERIM_DIR = DATA_DIR / "02_interim"
PROCESSED_DIR = DATA_DIR / "03_processed"

ENGLISH_DIR = RAW_DIR / "english_data"
JIGSAW_DIR_1 = ENGLISH_DIR / "jigsaw_toxic_comments"
JIGSAW_DIR_2 = ENGLISH_DIR / "jigsaw_unintended_bias"

JIGSAW_1_TRAIN_PATH = JIGSAW_DIR_1 / "train.csv"
JIGSAW_1_TEST_PATH = JIGSAW_DIR_1 / "test.csv"
JIGSAW_1_TEST_LABELS_PATH = JIGSAW_DIR_1 / "test_labels.csv"

JIGSAW_2_TRAIN_PATH = JIGSAW_DIR_2 / "train.csv"
# JIGSAW_2_TEST_PRIVATE_PATH = JIGSAW_DIR_2 / "test_private_expanded.csv"
# JIGSAW_2_TEST_PUBLIC_PATH = JIGSAW_DIR_2 / "test_public_expanded.csv"

# Other paths can be added here as needed