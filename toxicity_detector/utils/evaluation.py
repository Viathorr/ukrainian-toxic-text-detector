from typing import List

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    roc_auc_score,
    hamming_loss,
    average_precision_score
)

def create_compute_metrics_fn(labels_columns: List[str], threshold: float = 0.5, label_thresholds: List[float] = None):
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
        
        # Convert logits to probabilities
        probabilities = expit(predictions)
        
        # Apply thresholds
        if label_thresholds is not None:
            pred_binary = (probabilities >= label_thresholds).astype(int)
        else:
            pred_binary = (probabilities >= threshold).astype(int)

        # Aggregate metrics
        macro_f1 = f1_score(labels, pred_binary, average="macro", zero_division=0)
        micro_f1 = f1_score(labels, pred_binary, average="micro", zero_division=0)
        macro_precision = precision_score(labels, pred_binary, average="macro", zero_division=0)
        macro_recall = recall_score(labels, pred_binary, average="macro", zero_division=0)
        subset_accuracy = accuracy_score(labels, pred_binary)
        hamming_loss_score = hamming_loss(labels, pred_binary)  # the lower the better

        # ROC-AUC for each class
        try:
            roc_auc_scores = roc_auc_score(labels, probabilities, average=None)
        except ValueError:
            roc_auc_scores = [0.0] * len(labels_columns)

        # Precision-Recall AUC for each class
        try:
            pr_auc_scores = average_precision_score(labels, probabilities, average=None)
        except ValueError:
            pr_auc_scores = [0.0] * len(labels_columns)

        # Per-class F1
        per_class_f1 = f1_score(labels, pred_binary, average=None, zero_division=0)

        results = {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "subset_accuracy": subset_accuracy,
            "hamming_loss": hamming_loss_score
        }

        # Add per-class metrics
        for i, label in enumerate(labels_columns):
            results[f"{label}_f1"] = float(per_class_f1[i])
            results[f"{label}_roc_auc"] = float(roc_auc_scores[i])  
            results[f"{label}_pr_auc"] = float(pr_auc_scores[i])   

        return results

    return compute_metrics

def optimize_all_thresholds(probabilities, labels, label_columns, logger=None):
    """
    Optimize thresholds for each label to maximize F1 score.
    
    Parameters
    ----------
    probabilities : np.ndarray
        predicted probabilities for each label (shape: [N, num_labels])
    labels : np.ndarray
        true binary labels for each label (shape: [N, num_labels])
    label_columns : list    
        list of label names corresponding to the columns in probabilities and labels
    logger : logging.Logger, optional
        logger for logging optimization results, by default None
        
    Returns
    -------
    optimal_thresholds : dict
        dictionary mapping each label to its optimal threshold
    """
    optimal_thresholds = {}
    
    if logger:
        logger.info("Optimizing thresholds...")
        logger.info("=" * 80)
        logger.info(f"{'Label':<15} | {'Thresh@0.5':<10} | {'Optimal':<10} | {'F1@0.5':<10} | {'F1@Opt':<10} | {'Gain'}")
        logger.info("=" * 80)
    
    for i, label in enumerate(label_columns):
        # Current F1 at 0.5
        preds_05 = (probabilities[:, i] >= 0.5).astype(int)
        f1_05 = f1_score(labels[:, i], preds_05, zero_division=0)
        
        # Find optimal
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for thresh in thresholds:
            preds = (probabilities[:, i] >= thresh).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        optimal_thresholds[label] = best_threshold
        
        gain = best_f1 - f1_05
        
        if logger:
            logger.info(f"{label:<15} | {0.5:<10.2f} | {best_threshold:<10.2f} | "
                        f"{f1_05:<10.4f} | {best_f1:<10.4f} | {gain:+.4f}")
    
    return optimal_thresholds