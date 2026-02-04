import pandas as pd
from tqdm import tqdm
from toxicity_detector.config.paths import LOGS_DIR, JIGSAW_INTERIM, JIGSAW_PROCESSED
from toxicity_detector.utils.logger import setup_logger
from toxicity_detector.utils import text_cleaning as tc

tqdm.pandas()

# Set up logger
log_file_path = LOGS_DIR / "preprocess_jigsaw.log"
logger = setup_logger("preprocess_jigsaw", log_file=log_file_path)


def clean_jigsaw_for_translation(text: str) -> str:
    """Cleans Jigsaw text data for translation."""
    
    if not isinstance(text, str):
        return ""

    preprocessor = tc.TextPreprocessor(
        remove_wiki_elements=True,
        remove_nonprintable_characters=True,
        remove_repeated_sentences=True,
        remove_repeated_chunks=True
    )
    
    text = preprocessor.clean_text(text)
    
    return text

def preprocess_jigsaw_dataset():
    """Preprocess the Jigsaw dataset for translation and save the cleaned data."""
    
    for name, path in JIGSAW_INTERIM.items():
        logger.info(f"Processing {name} dataset")
        df = pd.read_csv(path)
        
        logger.info(f"Cleaning {len(df)} rows...")
        df["comment_text"] = df["comment_text"].progress_apply(clean_jigsaw_for_translation)
        
        initial_count = len(df)
        df = df[df["comment_text"].str.strip() != ""]
        logger.info(f"Removed {initial_count - len(df)} empty row(s).")
        
        output_path = JIGSAW_PROCESSED[name]
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")
    
    
if __name__ == "__main__":
    preprocess_jigsaw_dataset()