import pandas as pd
import torch
import torch.nn as nn
from transformers import Trainer


def calculate_class_weights(labels_df: pd.DataFrame, label_columns: list) -> torch.Tensor:
    """
    Calculate class weights for each label.
    
    Parameters
    ----------  
    labels_df : pd.DataFrame
        A DataFrame containing the labels for each comment.
    label_columns : list
        A list of label column names.
    
    Returns
    -------
    torch.Tensor
        A tensor of class weights.
    """
    pos_weights = []
    
    for i, label in enumerate(label_columns):
        pos_count = labels_df[label].sum()
        neg_count = len(labels_df) - pos_count
        
        # Calculate weight (neg / pos ratio)
        pos_weight = neg_count / (pos_count + 1e-5) 
        
        pos_weights.append(pos_weight)
        
        print(f"{label}:")
        print(f"  Positive: {pos_count} ({100 * pos_count / len(labels_df):.2f}%)")
        print(f"  Negative: {neg_count} ({100 * neg_count / len(labels_df):.2f}%)")
        print(f"  Weight: {pos_weight:.2f}")
    
    return torch.tensor(pos_weights, dtype=torch.float32)


# A custom Trainer for loss computation with pos_weights
class WeightedLossTrainer(Trainer):
    def __init__(self, pos_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.logits

        if self.pos_weights is not None:
            loss_fct = nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weights.to(model.device)
            )
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss