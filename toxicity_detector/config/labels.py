# Dataset labels in English (for model training and evaluation)
LABELS_EN = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COL = "comment"

# Primary label for binary classification
PRIMARY_LABEL = "toxic"

# Dataset labels in Ukrainian
LABELS_UK = {
    "toxic": "токсичний",
    "severe_toxic": "дуже токсичний",
    "obscene": "нецензурний",
    "threat": "погроза",
    "insult": "образа",
    "identity_hate": "мова ворожнечі"
}

def get_label_name(label: str, lang: str = "uk") -> str:
    """
    Get display name for a label.
    
    Parameters
    ----------
    label : str
        Label key in English.
    lang : str
        Language code ('ua' or 'en').
    
    Returns
    -------
    str
        Display name.
    """
    if lang == "uk":
        return LABELS_UK.get(label, label)
    return label.replace("_", " ").capitalize().title()