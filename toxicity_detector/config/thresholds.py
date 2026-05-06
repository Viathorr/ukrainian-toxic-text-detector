from toxicity_detector.config.labels import LABELS_EN

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
