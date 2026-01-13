import requests
import re
import json
import sys
import time

import pandas as pd
import typing_extensions as typing

from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.config.paths import JIGSAW_PROCESSED, LOGS_DIR
from toxicity_detector.config.settings import OPENROUTER_API_URL, OPENROUTER_HEADERS

# Set up logger
log_file_path = LOGS_DIR / "translate_jigsaw.log"
logger = setup_logger("translate_jigsaw", log_file=log_file_path)


class TranslationEntry(typing.TypedDict):
    id: str
    translated: str

prompt = """
Translate the following English comment into natural, fluent Ukrainian.

Rules:
- Preserve the original meaning, tone, and level of toxicity.
- Do NOT censor, soften, or intensify insults, threats, or hateful language.
- Do NOT add explanations or comments.
- Do NOT repeat phrases if they are clearly repetitive or accidental.
- Remove meaningless repetitions while preserving intent.
- Ignore formatting artifacts or noise.
- Output ONLY the translated Ukrainian text.

Example:

id: jdkewf2393, text: 'Zuma is a hopeless basket case. And he hates all white people, which clearly makes him a racist.'
Response: 
{
    \"id\": \"jdkewf2393\",
    \"translated\": \"Зума - безнадійний випадок. І він ненавидить усіх білих людей, що явно робить його расистом.\"
}

Now translate the following comments:
"""

def send_request(model_name, request):
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
    return response.json()
    

def translate_jigsaw_comments(model_name, df, batch_size=10, num_requests: int = 20) -> pd.DataFrame:
    comments = df[["id", "comment_text"]]
    
    translations: list[TranslationEntry] = []
    timeout = 2  # seconds between requests
    
    for i in range(0, num_requests):
        start = i * batch_size
        end = start + batch_size
        batch_comments = comments.iloc[start:end, :]
        
        if batch_comments.empty:
            break
        
        request_lines = []

        for _, comment in batch_comments.iterrows():
            line = f"id: {comment['id']}, text: '{comment['comment_text']}'"

            request_lines.append(line)
        
        logger.info(f"Prepared batch with {len(request_lines)} comments.")
            
        request_text = prompt + "\n".join(request_lines)
        logger.info(f"Sending request {i+1}/{num_requests} with {len(batch_comments)} comments")
        
        try:
            response = send_request(model_name, request_text)
            print(response)
            
            # The model's response is in response['choices'][0]['message']['content']
            response_text = response['choices'][0]['message']['content']
            response_chunks = response_text.strip().split("\n\n")
            
            # TODO: Change the logic of parsing based on actual response format
            # Regex to extract id and translated text
            pattern = r'id:\s*(?P<id>[^,]+),\s*translated:\s*"(?P<translated>.*)"'
            
            for chunk in response_chunks:
                try:
                    match = re.search(pattern, chunk)
                    
                    if match:
                        entry = match.groupdict()
                        translations.append(entry)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode chunk: {chunk}")
        except Exception as e:
            logger.error(f"Request failed: {e}")
            continue
        
        time.sleep(timeout)  # wait
        
    translated_df = pd.DataFrame(translations)
        
    return translated_df


if __name__ == "__main__":
    data_type = sys.argv[1]  # 'train' or 'test'
    
    # Load datasets
    en_path = JIGSAW_PROCESSED[data_type]
    en_df = pd.read_csv(en_path)
    
    ua_path = JIGSAW_PROCESSED[f"{data_type}_ua"]
    ua_df = pd.read_csv(ua_path) if ua_path.exists() else pd.DataFrame(columns=["id", "translated"])
    
    # Count how many comments need translation
    to_translate_df = en_df[~en_df["id"].isin(ua_df["id"])]
    
    if to_translate_df.empty:
        logger.info("No new comments to translate. Exiting.")
        sys.exit(0)

    logger.info(f"Translating {len(to_translate_df)} comments from {data_type} dataset")
    
    model_name = "mistralai/devstral-2512:free"
    logger.info(f"Using model: {model_name}")

    translated_df = translate_jigsaw_comments(model_name, to_translate_df, num_requests=10, batch_size=10)
    
    # Merge translations back into existing Ukrainian dataframe
    combined_df = pd.concat([ua_df, translated_df]).drop_duplicates(subset=["id"]).reset_index(drop=True)
    combined_df.to_csv(JIGSAW_PROCESSED[f"{data_type}_ua"], index=False)
    logger.info(f"Saved translated {data_type} dataset to {JIGSAW_PROCESSED[f'{data_type}_ua']}")