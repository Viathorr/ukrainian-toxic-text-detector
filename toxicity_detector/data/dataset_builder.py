from typing import List

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer


def create_dataset_from_csv(
    csv_path: str,
    text_column: str,
    label_columns: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    seed: int = 42
) -> Dataset:
    """
    Create HuggingFace Dataset from CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
    text_column : str
        Name of text column
    label_columns : List[str]
        List of label column names
    tokenizer : PreTrainedTokenizer
        Tokenizer for text encoding
    max_length : int
        Maximum sequence length
        
    Returns
    -------
    Dataset
        Prepared HuggingFace Dataset
    """
    df = pd.read_csv(csv_path)
    
    dataset = Dataset.from_pandas(df[[text_column] + label_columns])
    dataset = dataset.shuffle(seed=seed)
    
    preprocess_fn = create_preprocess_fn(
        tokenizer=tokenizer,
        text_column=text_column,
        label_columns=label_columns,
        max_length=max_length
    )
    
    # Apply preprocessing
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset

def create_preprocess_fn(tokenizer: PreTrainedTokenizer, text_column: str, label_columns: List[str], max_length: int):
    """
    Create preprocessing function for dataset mapping.
    
    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding text
    text_column : str
        Name of text column
    label_columns : List[str]
        List of label column names
    max_length : int
        Maximum sequence length
        
    Returns
    -------
    callable
        Preprocessing function
    """
    def preprocess_function(examples):
        """
        Preprocess batch of examples.
        
        Parameters
        ----------
        examples : Dict
            Batch of examples with text and labels
            
        Returns
        -------
        Dict
            Tokenized inputs with labels
        """
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
        )

        tokenized["labels"] = [
            [float(examples[label][i]) for label in label_columns]
            for i in range(len(examples[text_column]))
        ]

        return tokenized

    return preprocess_function