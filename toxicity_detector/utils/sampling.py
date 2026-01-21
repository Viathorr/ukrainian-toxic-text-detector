from typing import List, Tuple

import pandas as pd
import numpy as np


def select_comments_by_identity(df: pd.DataFrame, categories: list, n_samples: int) -> pd.DataFrame:
    """
    Selects comments from a DataFrame based on their identity categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing comments and their corresponding labels.
    categories : list
        List of identity categories to sample from.
    n_samples : int
        Total number of samples to select.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sampled comments, with the "keep" column dropped.
    """
    df = df.copy()
    df["keep"] = 0
    samples_per_category = n_samples // len(categories)
    
    for category in categories:
        category_comments = df[(df[category] == 1) & (df["comment_length"] <= 400)]
        category_comments = category_comments.sort_values(by="category_count", ascending=True)
        sampled_comments = category_comments.head(n=samples_per_category)
        df.loc[sampled_comments.index, "keep"] = 1
        
    return df[df["keep"] == 1].drop(columns=["keep"])

def rule_based_train_test_split(
    df: pd.DataFrame,
    toxicity_columns: List[str],
    identity_categories: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train-test split preserving toxic/identity ratios.
    
    This split ensures:
    - Original toxic:non-toxic ratio is preserved
    - Identity mention ratio is preserved within toxic/non-toxic groups
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing comments and their corresponding labels.
    toxicity_columns : List[str]
        List of toxicity columns to use for stratification.
    identity_categories : List[str]
        List of identity categories to use for stratification.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the train and test DataFrames.
    """
    np.random.seed(random_state)
    df = df.copy()

    # Identify toxic and identity comments
    is_toxic = df[toxicity_columns].any(axis=1)
    is_identity = df[identity_categories].any(axis=1)

    total_test_size = int(len(df) * test_size)

    # Preserve original toxic/non-toxic ratio
    toxic_ratio = is_toxic.mean()
    toxic_test_n = int(total_test_size * toxic_ratio)
    non_toxic_test_n = total_test_size - toxic_test_n

    # Preserve identity ratio within each group
    toxic_identity_ratio = is_identity[is_toxic].mean()
    non_toxic_identity_ratio = is_identity[~is_toxic].mean()

    toxic_identity_n = int(toxic_test_n * toxic_identity_ratio)
    toxic_plain_n = toxic_test_n - toxic_identity_n

    non_toxic_identity_n = int(non_toxic_test_n * non_toxic_identity_ratio)
    non_toxic_plain_n = non_toxic_test_n - non_toxic_identity_n

    # Sample from each group
    test_parts = [
        df[is_toxic & is_identity].sample(
            n=min(toxic_identity_n, (is_toxic & is_identity).sum()),
            random_state=random_state
        ),
        df[is_toxic & ~is_identity].sample(
            n=min(toxic_plain_n, (is_toxic & ~is_identity).sum()),
            random_state=random_state
        ),
        df[~is_toxic & is_identity].sample(
            n=min(non_toxic_identity_n, (~is_toxic & is_identity).sum()),
            random_state=random_state
        ),
        df[~is_toxic & ~is_identity].sample(
            n=min(non_toxic_plain_n, (~is_toxic & ~is_identity).sum()),
            random_state=random_state
        ),
    ]

    test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
    train_df = df.drop(index=test_df.index)

    return train_df, test_df

def balance_binary_classes(
    df: pd.DataFrame,
    label_column: str,
    method: str = "downsample",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Balance binary classes using downsampling or upsampling.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing binary labels.
    label_column : str
        Name of the binary label column.
    method : str, optional
        Downsampling ("downsample") or upsampling ("upsample"). Default is "downsample".
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with balanced binary classes.
    """
    positive = df[df[label_column] > 0]
    negative = df[df[label_column] == 0]
    
    if method == "downsample":
        # Downsample majority to match minority
        n_samples = min(len(positive), len(negative))
        positive_sampled = positive.sample(n=n_samples, random_state=random_state)
        negative_sampled = negative.sample(n=n_samples, random_state=random_state)
    elif method == "upsample":
        # Upsample minority to match majority
        n_samples = max(len(positive), len(negative))
        positive_sampled = positive.sample(
            n=n_samples, 
            replace=True, 
            random_state=random_state
        )
        negative_sampled = negative.sample(
            n=n_samples, 
            replace=True, 
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    balanced_df = pd.concat([positive_sampled, negative_sampled])
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)