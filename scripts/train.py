import os
import argparse
import random
from dataclasses import asdict

import numpy as np
import wandb
from huggingface_hub import login
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.utils.evaluation import create_compute_metrics_fn
from toxicity_detector.utils.training import calculate_class_weights, WeightedLossTrainer
from toxicity_detector.config.model_config import ModelConfig
from toxicity_detector.config.training_config import TrainingConfig
from toxicity_detector.data.dataset_builder import create_dataset_from_csv
from toxicity_detector.config.labels import LABELS_EN, TEXT_COL
from toxicity_detector.config.paths import (
    TRAIN_LOGS_DIR, 
    JIGSAW_PROCESSED, 
    OUTPUT_MODELS_DIR,
    UKR_PROCESSED,
    get_model_path
)
from toxicity_detector.config.settings import (
    WANDB_API_KEY, 
    WANDB_PROJECT, 
    WANDB_RUN_NAME_PREFIX, 
    HF_TOKEN, 
    HF_USERNAME, 
    HF_MODEL_PREFIX
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XLM-RoBERTa toxicity detection model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["jigsaw", "ukr"],
        default="jigsaw",
        help="Dataset to train on (default: jigsaw)"
    )
    parser.add_argument(
        "--version", 
        type=int,
        default=1,
        help="Model version (default: 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=False,
        help="Use class weights for imbalanced data (default: False)"
    )
    parser.add_argument(
        "--report-to",
        type=str,
        choices=["wandb", "none"],
        default="none",
        help="Report to (default: none)"
    )

    args = parser.parse_args()

    return args

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_environment(version: int, seed: int = 42, wandb_enabled: bool = False):
    """
    Set up the training environment, including logging and authentication.
    """
    log_file_path = TRAIN_LOGS_DIR / f"train_v{version}.log"
    logger = setup_logger("train", log_file=log_file_path)

    if WANDB_API_KEY and wandb_enabled:
        try:
            wandb.login(key=WANDB_API_KEY)
            os.environ["WANDB_PROJECT"] = WANDB_PROJECT
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"
            os.environ["WANDB_WATCH"] = "false"
            logger.info("WandB authenticated")
        except Exception as e:
            logger.warning(f"WandB login failed: {e}")
    else:
        logger.warning("WANDB_API_KEY not found or WandB logging disabled.")

    # HuggingFace authentication
    if HF_TOKEN:
        try:
            login(token=HF_TOKEN)
            logger.info("HuggingFace authenticated")
        except Exception as e:
            logger.warning(f"HF login failed: {e}")
    else:
        logger.warning("HF_TOKEN not found.")
    
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    return logger   

def load_data(dataset: str, tokenizer: AutoTokenizer, max_length: int, seed: int = 42):
    """
    Load and preprocess datasets.
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load ("jigsaw" or "ukr")
    tokenizer : AutoTokenizer
        Tokenizer for text encoding
    max_length : int
        Maximum sequence length for tokenization
    seed : int, optional
        Random seed for shuffling (default: 42)
        
    Returns
    -------
    tuple
        A tuple containing the training, validation, and test datasets
    """
    train_path, val_path = None, None
    
    if dataset == "jigsaw":
        train_path = JIGSAW_PROCESSED["train_uk_bin"]
        val_path = JIGSAW_PROCESSED["test_uk_bin"]
    elif dataset == "ukr":
        train_path = UKR_PROCESSED["train"]
        val_path = UKR_PROCESSED["val"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    train_dataset = create_dataset_from_csv(
        train_path,
        text_column=TEXT_COL,
        label_columns=LABELS_EN,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed
    )

    val_dataset = create_dataset_from_csv(
        val_path,
        text_column=TEXT_COL,
        label_columns=LABELS_EN,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed
    )

    return train_dataset, val_dataset
    
   
def main():    
    args = parse_args()
    
    VERSION = args.version
    
    logger = setup_environment(version=VERSION, seed=args.seed, wandb_enabled=(args.report_to == "wandb"))
    
    # Paths
    model_save_dir = get_model_path(version=VERSION, dataset=args.dataset)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    logging_dir = TRAIN_LOGS_DIR / f"v{VERSION}"
    logging_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = OUTPUT_MODELS_DIR / f"v{VERSION}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    hub_model_id = f"{HF_USERNAME}/{HF_MODEL_PREFIX}-v{VERSION}"
    wandb_run_name = f"{WANDB_RUN_NAME_PREFIX}-v{VERSION}" if args.report_to == "wandb" else None
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    logger.info("Configuration loaded.")
    
    logger.info("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=model_config.num_labels,
        problem_type=model_config.problem_type
    )
    
    # Freeze encoder layers if specified
    if model_config.freeze_encoder_layers > 0:
        logger.info(f"Freezing bottom {model_config.freeze_encoder_layers} encoder layers...")
        
        # Freeze embeddings
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i in range(model_config.freeze_encoder_layers):
            for param in model.base_model.encoder.layer[i].parameters():
                param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Trainable: {trainable_params:,} / {total_params:,} "
            f"({100*trainable_params/total_params:.1f}%)"
        )

    # Load datasets
    logger.info(f"Loading {args.dataset} datasets...")
    train_dataset, val_dataset = load_data(
        dataset=args.dataset,
        tokenizer=tokenizer,
        max_length=model_config.max_seq_length,
        seed=args.seed
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        run_name=wandb_run_name,
        hub_model_id=hub_model_id,
        seed=args.seed,
        report_to=args.report_to,
        **asdict(training_config)
    )
    
    compute_metrics = create_compute_metrics_fn(LABELS_EN)
    
    # Trainer
    if args.use_class_weights:
        pos_weights = calculate_class_weights(train_dataset, LABELS_EN)
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            pos_weights=pos_weights
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(model_save_dir)
    
    logger.info("Model training complete.")
    
    
if __name__ == "__main__":
    main()