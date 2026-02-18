import argparse

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.utils.evaluation import create_compute_metrics_fn
from toxicity_detector.data.dataset_builder import create_dataset_from_csv
from toxicity_detector.config.thresholds import get_threshold_list
from toxicity_detector.config.labels import LABELS_EN, TEXT_COL
from toxicity_detector.config.paths import (
    EVAL_LOGS_DIR,
    UKR_PROCESSED,
    JIGSAW_PROCESSED,
    get_model_path
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate XLM-RoBERTa toxicity detection model on test set"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["jigsaw", "ukr"],
        default="jigsaw",
        help="Dataset the model was trained on (default: jigsaw)"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        choices=["jigsaw", "ukr"],
        default="ukr",
        help="Dataset to evaluate on (default: ukr)"
    )
    parser.add_argument(
        "--version", 
        type=int,
        default=1,
        help="Model version (default: 1)"
    )
    parser.add_argument(
        "--use-thresholds",
        action="store_true",
        help="Use optimized per-label thresholds instead of default 0.5"
    )

    args = parser.parse_args()

    return args    

def load_test_data(dataset: str, tokenizer, max_length: int = 128):
    """
    Load test dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name (jigsaw or ukr)
    tokenizer : AutoTokenizer
        Tokenizer
    max_length : int
        Maximum sequence length
        
    Returns
    -------
    Dataset
        Test dataset
    """
    if dataset == "jigsaw":
        test_path = JIGSAW_PROCESSED["test_uk_bin"]  # It's actually a dataset used for validation during training
    elif dataset == "ukr":
        test_path = UKR_PROCESSED["test"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    test_dataset = create_dataset_from_csv(
        csv_path=test_path,
        text_column=TEXT_COL,
        label_columns=LABELS_EN,
        tokenizer=tokenizer,
        max_length=max_length,
        remove_columns=False
    )
    
    return test_dataset
   

def main():    
    args = parse_args()
    
    VERSION = args.version
    
    log_file_path = EVAL_LOGS_DIR / f"evaluate_v{VERSION}.log"
    logger = setup_logger("evaluate", log_file=log_file_path)
    
    # Paths
    model_path = get_model_path(version=VERSION, dataset=args.dataset)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load test dataset
    logger.info(f"Loading test dataset...")
    test_dataset = load_test_data(dataset=args.test_dataset, tokenizer=tokenizer)
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    logger.info("Starting evaluation...")
    
    if args.use_thresholds:
        logger.info("Using optimized thresholds for evaluation.")
        
        threshold_list = get_threshold_list(label_names=LABELS_EN)
        compute_metrics = create_compute_metrics_fn(LABELS_EN, label_thresholds=threshold_list)
    else:
        logger.info("Using default threshold of 0.5 for evaluation.")
        compute_metrics = create_compute_metrics_fn(LABELS_EN)  # uses default 0.5 thresholds
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32),
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Evaluate
    results = trainer.evaluate()
    
    logger.info("=" * 70)
    logger.info("Evaluation Results")
    logger.info("=" * 70)
    
    metrics_to_log = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "subset_accuracy",
        "hamming_loss"
    ]

    for metric in metrics_to_log:
        logger.info(
            f"Test {metric.replace('_', ' ').title():<20}: "
            f"{results.get(f'eval_{metric}', 0):.4f}"
        )

    logger.info("=" * 70)
    logger.info("Per-Class F1 and AUC Scores:")
    logger.info("-" * 70)

    for label in LABELS_EN:
        logger.info(
            f"{label:<15}: "
            f"F1={results.get(f'eval_{label}_f1', 0):.4f}, "
            f"AUC={results.get(f'eval_{label}_roc_auc', 0):.4f}, "
            f"PR-AUC={results.get(f'eval_{label}_pr_auc', 0):.4f}"
        )

    logger.info("=" * 70)
    logger.info("Model evaluation complete.")
    
    
if __name__ == "__main__":
    main()
    