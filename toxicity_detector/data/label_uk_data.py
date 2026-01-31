import re

import pandas as pd

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.config.paths import UKR_RAW, LOGS_DIR
from toxicity_detector.config.labels import LABELS_EN
from toxicity_detector.config.vocabulary import OBSCENE_PATTERNS_UK, INSULTS_UK, INSULT_PATTERNS_UK, IDENTITY_SLUR_PATTERNS_UK, IDENTITY_SLURS_UK, THREAT_PATTERNS_UK

log_file_path = LOGS_DIR / "label_uk_data.log"
logger = setup_logger("label_uk_data", log_file=log_file_path)

# Labelling functions

def lf_is_toxic(df_row: pd.Series) -> int:
    """
    Labels a comment as toxic if it contains any of the following labels:

    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate

    Parameters
    ----------
    df_row : pd.Series
        A single row of a DataFrame containing comments and their corresponding labels.

    Returns
    -------
    bool
        True if the comment is toxic, False otherwise.
    """
    if df_row[LABELS_EN[1:]].sum() > 0:
        return 1
    else:
        return 0
    
def lf_is_severe_toxic(df_row: pd.Series) -> int:
    """
    Labels a comment as severe toxic if it contains the label "obscene" and at least one of the following labels:

    - threat
    - insult
    - identity hate

    Parameters
    ----------
    df_row : pd.Series
        A single row of a DataFrame containing comments and their corresponding labels.

    Returns
    -------
    bool
        True if the comment is severe toxic, False otherwise.
    """
    if df_row["obscene"] == 1 and (df_row["threat"] == 1 or df_row["insult"] == 1 or df_row["identity_hate"] == 1):
        return 1
    else:
        return 0
    
def contains_vocab(df_row: pd.Series, vocab: list = None, patterns: list = None) -> int:
    """
    Checks if a comment contains any of the given vocabulary or matches any of the given regular expression patterns.

    Parameters
    ----------
    df_row : pd.Series
        A single row of a DataFrame containing comments and their corresponding labels.
    vocab : list, optional
        A list of words to check for in the comment. Defaults to None.
    patterns : list, optional
        A list of regular expression patterns to match in the comment. Defaults to None.

    Returns
    -------
    int
        1 if the comment contains any of the given vocabulary or matches any of the given patterns, 0 otherwise.
    """
    if vocab:
        for word in vocab:
            if word in df_row["comment"].lower():
                return 1    
    
    if patterns:
        for pattern in patterns:
            if re.search(pattern, df_row["comment"].lower()):
                return 1
    
    return 0
    
def lf_is_obscene(df_row: pd.Series) -> int:
    return contains_vocab(df_row, patterns=OBSCENE_PATTERNS_UK)
    
def lf_is_insult(df_row: pd.Series) -> int:
    return contains_vocab(df_row, INSULTS_UK, INSULT_PATTERNS_UK)

def lf_is_identity_hate(df_row: pd.Series) -> int:
    return contains_vocab(df_row, IDENTITY_SLURS_UK, IDENTITY_SLUR_PATTERNS_UK)

def lf_is_threat(df_row: pd.Series) -> int:
    return contains_vocab(df_row, patterns=THREAT_PATTERNS_UK)

def prelabel_uk_dataset(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-labels a Ukrainian comments dataset using weak supervision labelling functions.

    Parameters
    ----------
    comments_df : pd.DataFrame
        A DataFrame containing comments and their IDs.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the labels for each comment.
    """
    labels_df = pd.DataFrame(index=comments_df["id"], columns=LABELS_EN)
    
    for _, row in comments_df.iterrows():
        labels_df.loc[row["id"], "obscene"] = lf_is_obscene(row)
        labels_df.loc[row["id"], "insult"] = lf_is_insult(row)
        labels_df.loc[row["id"], "identity_hate"] = lf_is_identity_hate(row)
        labels_df.loc[row["id"], "threat"] = lf_is_threat(row)
        labels_df.loc[row["id"], "severe_toxic"] = lf_is_severe_toxic(labels_df.loc[row["id"]])
        labels_df.loc[row["id"], "toxic"] = lf_is_toxic(labels_df.loc[row["id"]])
        
    return labels_df


def main():
    logger.info("Labelling UK comments...")
    comments_df = pd.read_csv(UKR_RAW["combined"])
    
    labels_df = prelabel_uk_dataset(comments_df)
    
    labels_df.to_csv(UKR_RAW["weak_labels"])
    
    logger.info(f"Saved {len(labels_df)} labels to {UKR_RAW['weak_labels']}")
    logger.info(labels_df.describe())
    
    
if __name__ == "__main__":
    main()
    
    