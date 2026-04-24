import torch
from scipy.special import expit
from transformers import PreTrainedTokenizer, PreTrainedModel

from toxicity_detector.data.text_cleaning import TextPreprocessor
from toxicity_detector.config.thresholds import get_threshold_list
from toxicity_detector.config.model_config import ModelConfig
from toxicity_detector.config.labels import LABELS_EN

preprocessor = TextPreprocessor(lowercase=True)  # Text preprocessor instance to clean and preprocess input text
model_config = ModelConfig()  # Model configuration instance to access max sequence length 


def predict(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> dict[str, float]:
    """
    Predict the toxicity probabilities for a given input text using the provided model and tokenizer.
    
    Parameters:
    -----------
    text : str
        The input text to predict toxicity probabilities for.
    model : transformers.PreTrainedModel
        The model to use for prediction.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for tokenization.
    
    Returns:
    --------
    dict[str, float]
        A dictionary mapping toxicity labels to their corresponding probabilities.
    """
    device = next(model.parameters()).device  
    
    # Preprocess the input text
    cleaned_text = preprocessor.clean_text(text)
    
    # Tokenize the input text
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=model_config.max_seq_length
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.inference_mode():
        outputs = model(**inputs)
        
    logits = outputs.logits.squeeze().cpu().numpy()
    probabilities = expit(logits)  # Apply sigmoid to get probabilities
    
    return {label: round(float(prob), 4) for label, prob in zip(LABELS_EN, probabilities)}


def predict_with_thresholds(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, thresholds: list[float] | None = None) -> dict[str, int]:
    """
    Predict the toxicity labels for a given input text using the provided model and tokenizer,
    applying the specified thresholds to convert probabilities to binary predictions.
    
    Parameters:
    -----------
    text : str
        The input text to predict toxicity labels for.
    model : transformers.PreTrainedModel
        The model to use for prediction.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use for tokenization.
    thresholds : list[float], optional
        A list of thresholds to apply to convert probabilities to binary predictions.
    
    Returns:
    --------
    dict[str, int]
        A dictionary mapping toxicity labels to their corresponding binary predictions.
    """
    if thresholds is None:
        thresholds = get_threshold_list()
        
    probabilities = predict(text, model, tokenizer)
    
    return {label: int(probabilities[label] >= threshold) for label, threshold in zip(LABELS_EN, thresholds)}