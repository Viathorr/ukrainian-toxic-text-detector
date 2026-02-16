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
        max_length=max_length
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
    
    compute_metrics = create_compute_metrics_fn(LABELS_EN)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=32),
        data_collator=data_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
        
    logger.info("Starting evaluation...")
    results = trainer.evaluate()
    
    logger.info("=" * 70)
    logger.info("Evaluation Results")
    logger.info("=" * 70)
    logger.info(f"Test Macro F1:       {results.get('eval_macro_f1', 0):.4f}")
    logger.info(f"Test Micro F1:       {results.get('eval_micro_f1', 0):.4f}")
    logger.info(f"Test Macro Precision: {results.get('eval_macro_precision', 0):.4f}")
    logger.info(f"Test Macro Recall:    {results.get('eval_macro_recall', 0):.4f}")
    logger.info(f"Test Subset Accuracy: {results.get('eval_subset_accuracy', 0):.4f}")
    logger.info("=" * 70)
    
    logger.info("\nPer-Class F1 and AUC Scores:")
    logger.info("-" * 70)
    
    for label in LABELS_EN:
        f1_score = results.get(f"eval_{label}_f1", 0)
        auc_score = results.get(f"eval_{label}_auc", 0)
        logger.info(f"{label:<15}: F1={f1_score:.4f}, AUC={auc_score:.4f}")
    
    logger.info("=" * 70)
    
    logger.info("Model evaluation complete.")
    
    
if __name__ == "__main__":
    main()
    