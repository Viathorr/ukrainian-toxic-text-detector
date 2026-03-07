import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.data.dataset_builder import create_dataset_from_csv
from toxicity_detector.config.thresholds import MARGIN_THRESHOLD
from toxicity_detector.config.labels import LABELS_EN, TEXT_COL
from toxicity_detector.config.paths import (
    LOGS_DIR,
    UKR_RAW,
    UKR_INTERIM,
    get_model_path
)


log_file_path = LOGS_DIR / "model_assisted_uk_data_labeling.log"
logger = setup_logger("label_uk_data", log_file=log_file_path)


def create_uncertainty_dataset(dataset: Dataset, logits: np.ndarray, label_columns: List[str] = LABELS_EN, margin_threshold: float = MARGIN_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates two datasets from a model's predictions on a given dataset.
    The first dataset contains the predicted labels for each comment in the dataset.
    The second dataset contains the predicted probabilities for each label for comments with uncertain predictions (i.e. predictions with a probability margin less than the given margin threshold).
    
    Parameters
    ----------
    dataset : Dataset
        The dataset for which the model has made predictions.
    logits : np.ndarray
        The model's predictions on the dataset, in logit format.
    label_columns : List[str]
        The column names for the predicted labels and probabilities datasets.
    margin_threshold : float
        The probability margin threshold above which a prediction is considered uncertain.

    Returns
    -------
    labels_df : pd.DataFrame
        A dataframe containing the predicted labels for each comment in the dataset.
    uncertain_df : pd.DataFrame
        A dataframe containing the predicted probabilities for each label for comments with uncertain predictions.
    """
    ids = dataset["id"]
    comments = dataset["comment"]
    
    probs = expit(logits)  # convert logits to probabilities (shape: [N, num_labels])
    preds = (probs > 0.5).astype(int)  # convert probabilities to predictions (shape: [N, num_labels])
    
    # Create a new dataframe with predicted labels
    labels_df = pd.DataFrame(data=preds, columns=label_columns)
    labels_df.insert(0, "id", ids)
    labels_df.insert(1, "comment", comments)
    
    # Create a new dataframe with uncertain prediction probabilities
    margins = np.abs(probs - 0.5)
    min_margins = margins.min(axis=1)
    
    uncertain_mask = min_margins < margin_threshold
    
    uncertain_probs = probs[uncertain_mask]
    uncertain_ids = np.array(ids)[uncertain_mask]  
    
    uncertain_df = pd.DataFrame(data=uncertain_probs, columns=label_columns)
    uncertain_df.insert(0, "id", uncertain_ids)
    
    return labels_df, uncertain_df  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=int, default=12)
    args = parser.parse_args()
    
    logger.info(f"Model assisted data labeling for version {args.model_version}...")

    logger.info("Loading model and tokenizer...")

    model_path = get_model_path(args.model_version, dataset="jigsaw")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    logger.info("Loading dataset...")
    dataset = create_dataset_from_csv(csv_path=UKR_RAW["combined"], id_column="id", text_column=TEXT_COL, tokenizer=tokenizer, label_columns=[], max_length=128, remove_columns=False)
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32),
        data_collator=data_collator,
        compute_metrics=None,
    )
    
    preds = trainer.predict(dataset)
    
    logits = preds.predictions
    labels_df, uncertain_df = create_uncertainty_dataset(dataset, logits)
    
    logger.info("Saving data...")
    labels_df.to_csv(UKR_INTERIM["labels"], index=False)
    uncertain_df.to_csv(UKR_INTERIM["probs"], index=False)
    
    logger.info("Data labeling completed.")
    
    
if __name__ == "__main__":
    main()