from typing import List

from scipy.special import expit
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    roc_auc_score
)

def create_compute_metrics_fn(labels_columns: List[str], threshold: float = 0.5):
    """
    Creates a function to compute evaluation metrics for multi-label classification.

    Parameters
    ----------
    labels : list
        list of labels for multi-label classification
    threshold : float
        threshold for converting probabilities to binary predictions, defaults to 0.5

    Returns
    -------
    callable
        Compute metrics function that can be used with Hugging Face Trainer for multi-label classification tasks.
    """
    def compute_metrics(eval_pred):
        """
        Compute evaluation metrics for multi-label classification.
        Assumes labels shape (N, num_labels).
        
        Parameters
        ----------
        eval_pred : tuple
            tuple containing the predictions and labels for evaluation
        
        Returns
        -------
        dict
            a dictionary containing the computed evaluation metrics
            - macro_f1: macro F1 score
            - macro_precision: macro precision score
            - macro_recall: macro recall score
            - micro_f1: micro F1 score
            - subset_accuracy: subset accuracy score
            - <label>_f1: per-class F1 score for each label in labels_columns
            - <label>_auc: per-class AUC score for each label in labels_columns
        """
        # Assumes multi-label classification with shape (N, num_labels)
        predictions, labels = eval_pred
        
        # Convert to probabilities and binary predictions
        probabilities = expit(predictions)  
        
        pred_binary = (probabilities >= threshold).astype(int)

        macro_f1 = f1_score(labels, pred_binary, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, pred_binary, average="micro", zero_division=0)
        macro_precision = precision_score(labels, pred_binary, average="macro", zero_division=0)
        macro_recall = recall_score(labels, pred_binary, average="macro", zero_division=0)

        subset_accuracy = accuracy_score(labels, pred_binary)

        try:
            per_class_auc = roc_auc_score(labels, probabilities, average=None)
        except ValueError:
            per_class_auc = [0.0] * len(labels_columns)

        per_class_f1 = f1_score(labels, pred_binary, average=None, zero_division=0)

        results = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "subset_accuracy": subset_accuracy,
        }

        for i, label in enumerate(labels_columns):
            results[f"{label}_f1"] = per_class_f1[i]
            results[f"{label}_auc"] = per_class_auc[i]

        return results

    return compute_metrics