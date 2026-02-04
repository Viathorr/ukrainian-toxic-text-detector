import sys
import argparse

import pandas as pd

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.config.paths import JIGSAW_PROCESSED, UKR_INTERIM, UKR_PROCESSED, LOGS_DIR
from toxicity_detector.utils.text_cleaning import TextPreprocessor

# Set up logger
log_file_path = LOGS_DIR / "preprocess_dataset.log"
logger = setup_logger("preprocess_dataset", log_file=log_file_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    
    parser.add_argument(
        "--data-name",
        choices=["train", "test", "train_uk", "test_uk", "train_uk_bin", "test_uk_bin"],
        default="train",
        help="Data type to preprocess (default: train)"
    )
    parser.add_argument(
        "--data-kind",
        choices=["jigsaw", "ukr"],
        default="ukr",
        help="Data kind to preprocess (default: ukr)"
    )
    
    return parser.parse_args()

def preprocess_dataset(data_name, data_kind):
    preprocessor = TextPreprocessor()

    logger.info(f"Preprocessing {data_kind} {data_name} data...")

    try:
        if data_kind == "jigsaw":
            path = JIGSAW_PROCESSED[data_name]
            data = pd.read_csv(path)

            # Jigsaw comments were cleaned prior to translation;
            # this step applies post-translation normalization to ensure consistency
            data["translated"] = data["translated"].apply(preprocessor.clean_text)
            data.to_csv(JIGSAW_PROCESSED[data_name], index=False)
            logger.info(f"Saved to {JIGSAW_PROCESSED[data_name]}")
        elif data_kind == "ukr":
            path = UKR_INTERIM[data_name]
            data = pd.read_csv(path)

            data["comment"] = data["comment"].apply(preprocessor.clean_text)
            data.to_csv(UKR_PROCESSED[data_name], index=False)
            logger.info(f"Saved to {UKR_PROCESSED[data_name]}")
        else:
            raise ValueError(f"Invalid data kind: {data_kind}")

    except Exception as e:
        logger.error(f"Error preprocessing {data_kind} {data_name} data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    
    preprocess_dataset(args.data_name, args.data_kind)