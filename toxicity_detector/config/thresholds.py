from toxicity_detector.config.labels import LABELS_EN

# Uncertainty margin threshold
MARGIN_THRESHOLD = 0.15  # |p_pred - 0.5| < MARGIN_THRESHOLD -> uncertain

# Prediction thresholds
DEFAULT_THRESHOLDS = {
    "toxic": 0.5,
    "severe_toxic": 0.4,
    "obscene": 0.6,
    "threat": 0.6,
    "insult": 0.5,
    "identity_hate": 0.4
}

def get_threshold_list(label_names=LABELS_EN):
    return [DEFAULT_THRESHOLDS[label] for label in label_names]
