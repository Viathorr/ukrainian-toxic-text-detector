import sys
import re
import json
import time
import math
import logging
import requests
import argparse

import pandas as pd

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.config.paths import JIGSAW_PROCESSED, LOGS_DIR
from toxicity_detector.config.settings import OPENROUTER_API_URL, OPENROUTER_HEADERS

# Set up logger
log_file_path = LOGS_DIR / "translate_jigsaw.log"
logger = setup_logger("translate_jigsaw", log_file=log_file_path, level=logging.DEBUG)

# Translation prompt template
TRANSLATION_PROMPT = """
Translate the following English comment into natural, fluent Ukrainian.

Rules:
- Preserve the original meaning, tone, and level of toxicity.
- Do NOT censor, soften, or intensify insults, threats, or hateful language.
- Do NOT add explanations or comments.
- Do NOT repeat phrases if they are clearly repetitive or accidental.
- Remove meaningless repetitions while preserving intent.
- Ignore formatting artifacts or noise.
- Output ONLY the translated Ukrainian text.

Additional output constraints:
- The translated text MUST NOT contain double quotes (").
- If quotes are required, use single quotation marks (').
- Do NOT escape quotes.
- Do NOT add line breaks.
- Each translation must be on a single line.

You MUST return a JSON array.
Each element MUST be an object with exactly two fields:
- id (string)
- translated (string)

Return ONLY valid JSON.
No prose. No comments.

Now translate the following comments:
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-translate comments with an LLM"
    )

    parser.add_argument(
        "--data-type",
        choices=["train", "test"],
        default="train",
        help="Data type to translate (default: train)"
    )
    parser.add_argument(
        "--model", 
        default="mistralai/devstral-2512:free",
        help="Model name (default: mistralai/devstral-2512:free)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Maximum number of batch requests to send (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of comments per batch (default: 20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Seconds to wait between requests (default: 5)"
    )

    args = parser.parse_args()

    if args.batch_size > 50:
        parser.error("batch_size > 50 risks truncated model output")

    return args

def sanitize_translation(text: str) -> str:
    """
    Sanitize the translation text according to specified rules.

    Parameters
    ----------
    text : str
        The translation text to sanitize.

    Returns
    -------
    str
        Sanitized translation text.
    """
    if not text:
        return text

    text = text.replace('"', "'")  # Replace all double quotes with single quotes
    text = text.replace('\n', ' ').replace('\r', ' ')  # Ensure single line
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace

    return text

def strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences from text.

    Parameters
    ----------
    text : str
        Text potentially wrapped in ```

    Returns
    -------
    str
        Text without code fences.
    """
    text = text.strip()
    
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        
    return text.strip()

def send_request(model_name: str, request: str) -> dict:
    """
    Send a request to the OpenRouter API.
    
    Parameters
    ----------
    model_name : str
        Name of the model to use.
    request : str
        The text of the request.
    
    Returns
    -------
    dict
        The response from the OpenRouter API.
    """
    response = requests.post(
        url=OPENROUTER_API_URL,
        headers=OPENROUTER_HEADERS,
        data=json.dumps({
            "model": model_name,
            "messages": [
                {"role": "user", "content": request}
            ]
        })
    )
    response.raise_for_status()
    
    return response.json()
    
def parse_response(response_text: str) -> list[dict]:
    """
    Parse API response text into list of translation dicts.
    
    Parameters
    ----------
    response_text : str
        The response text from the OpenRouter API.
    
    Returns
    -------
    list[dict]
        List of translation dicts. Each dict has "id" and "translated" keys.
    """
    response_text = strip_code_fences(response_text)
    translations = []
    
    # Try JSON parsing first
    try:
        data = json.loads(response_text)

        if isinstance(data, list):
            for obj in data:
                if "id" in obj and "translated" in obj:
                    translations.append({
                        "id": str(obj["id"]).strip(),
                        "translated": sanitize_translation(obj["translated"])
                    })
            return translations

    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response, attempting regex parsing")
    
    # Fallback to regex parsing
    pattern = re.compile(
        r"""
        \bid\b\s*[:=]\s*               # "id" with word boundary, followed by : or =
        ["']?(?P<id>[a-zA-Z0-9_-]+)["']?     # Capture ID (with optional quotes)
        .*?                            # Skip to translated field
        \btranslated\b\s*[:=]\s*       # "translated" with word boundary
        ["']?                          # Optional opening quote
        (?P<translated>.+?)                          # Capture translation (non-greedy)
        ["']?                          # Optional closing quote
        (?=\s*[,}\]]|$)                # Look ahead: comma, brace, bracket, or end
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    for match in pattern.finditer(response_text):
        translations.append({
            "id": match.group("id").strip(),
            "translated": sanitize_translation(match.group("translated").strip())
        })
    
    return translations

def format_batch_request(batch_comments: pd.DataFrame) -> str:
    """
    Format a batch of comments into a request string.
    
    Parameters
    ----------
    batch_comments : pd.DataFrame
        DataFrame containing the comments to translate. Must have 'id' and 'comment_text' columns.
    
    Returns
    -------
    str
        Formatted request string.
    """
    request_lines = []
    
    for _, comment in batch_comments.iterrows():
        line = f"id: {comment['id']}, text: '{comment['comment_text']}'"
        request_lines.append(line)
    
    return TRANSLATION_PROMPT + "\n".join(request_lines)

def translate_jigsaw_comments(
    model_name: str,
    df: pd.DataFrame,
    batch_size: int,
    num_requests: int,
    timeout: int
    ) -> pd.DataFrame:
    comments = df[["id", "comment_text"]]
    
    translations = []  # List of dicts with "id" and "translated"
    
    total_batches = math.ceil(len(comments) / batch_size)
    num_batches = min(num_requests, total_batches)
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_comments = comments.iloc[start:end, :]
        
        if batch_comments.empty:
            break
        
        request_text = format_batch_request(batch_comments)
        logger.info(f"Sending request {i+1}/{num_requests} with {len(batch_comments)} comments")
        
        try:
            response = send_request(model_name, request_text)
            
            response_text = response["choices"][0]["message"]["content"]
            batch_translations = parse_response(response_text)
            
            # Validate we got translations
            if not batch_translations:
                logger.error(f"No translations parsed from batch {i+1}")
                logger.debug(f"Raw response: {response}")
                continue
                
            # Validate IDs match
            requested_ids = set(batch_comments["id"].astype(str))
            received_ids = {t["id"] for t in batch_translations}
            missing_ids = requested_ids - received_ids
            
            if missing_ids:
                logger.warning(f"Missing translations for IDs: {missing_ids}")
            
            translations.extend(batch_translations)
            logger.info(f"Successfully translated {len(batch_translations)} comments from batch {i+1}")
            
        except Exception as e:
            logger.error(f"Batch {i+1} translation failed: {e}")
            continue
        
        # Add a delay between requests except after the last one
        if i < num_batches - 1:
            time.sleep(timeout) 
        
    logger.info(f"Total translations collected: {len(translations)}")
    
    return pd.DataFrame(translations)

def main():
    args = parse_args()
    
    logger.info(f"Using model \"{args.model}\" to translate {args.data_type} dataset")
    logger.info(f"Parameters: num_requests={args.num_requests}, batch_size={args.batch_size}, timeout={args.timeout}")
    
    # Load datasets
    en_path = JIGSAW_PROCESSED[args.data_type]
    en_df = pd.read_csv(en_path)
    logger.info(f"Loaded {len(en_df)} comments from {en_path}")
    
    ua_path = JIGSAW_PROCESSED[f"{args.data_type}_ua"]
    ua_df = pd.read_csv(ua_path) if ua_path.exists() else pd.DataFrame(columns=["id", "translated"])
    logger.info(f"Loaded {len(ua_df)} existing translations from {ua_path}")
    
    # Count how many comments need translation
    to_translate_df = en_df[~en_df["id"].isin(ua_df["id"])]
    
    if to_translate_df.empty:
        logger.info("No new comments to translate. Exiting.")
        sys.exit(0)

    logger.info(f"{len(to_translate_df)} comments need translation from {args.data_type} dataset")

    # Translate
    translated_df = translate_jigsaw_comments(
        model_name=args.model,
        df=to_translate_df,
        num_requests=args.num_requests,
        batch_size=args.batch_size,
        timeout=args.timeout
    )
    
    if translated_df.empty:
        logger.error("No translations were successful. Exiting without saving.")
        sys.exit(1)
    
    # Merge translations back into existing Ukrainian dataframe
    combined_df = pd.concat([ua_df, translated_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    combined_df.to_csv(ua_path, index=False)
    
    logger.info(f"Saved {len(translated_df)} new translations to {ua_path}")
    logger.info(f"Total translations in file: {len(combined_df)}")


if __name__ == "__main__":
    main()