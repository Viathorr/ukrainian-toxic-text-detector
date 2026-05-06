import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from toxicity_detector.config.settings import HF_USERNAME, HF_MODEL_PREFIX, HF_MODEL_VERSION


def load_model(model_version: int = HF_MODEL_VERSION) -> tuple:
    """
    Load a Hugging Face model and tokenizer.

    Parameters:
    -----------
    model_version : int, optional
        The version of the model to load, by default HF_MODEL_VERSION.

    Returns:
    --------
    tuple
        A tuple containing the loaded model and tokenizer.
    """
    model_path = f"{HF_USERNAME}/{HF_MODEL_PREFIX}-v{model_version}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer