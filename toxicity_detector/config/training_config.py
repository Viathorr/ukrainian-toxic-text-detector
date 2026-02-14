from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training the toxicity detection model."""

    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    num_train_epochs: int = 5

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64

    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"

    push_to_hub: bool = True
    hub_private_repo: bool = True
    hub_strategy: str = "every_save"

    logging_strategy: str = "steps"
    logging_steps: int = 500

    report_to: str = "wandb"
    metric_for_best_model: str = "macro_f1"
    load_best_model_at_end: bool = True
