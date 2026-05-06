from dataclasses import dataclass

from toxicity_detector.config.labels import LABELS_EN

@dataclass
class ModelConfig:
    """Configuration for the toxicity detection model."""

    model_name_or_path: str = "FacebookAI/xlm-roberta-base"
    num_labels: int = len(LABELS_EN)
    problem_type: str = "multi_label_classification"
    
    freeze_encoder_layers: int = 0  # Number of encoder layers to freeze (0 means no freezing)
    
    max_seq_length: int = 128  # Maximum sequence length for tokenization