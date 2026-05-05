import dotenv
from toxicity_detector.config.paths import ROOT_DIR

ENV_FILE = ROOT_DIR / ".env"
dotenv.load_dotenv(ENV_FILE)

# API Keys and Tokens
OPENROUTER_API_KEY = dotenv.get_key(ENV_FILE, "OPENROUTER_API_KEY")
WANDB_API_KEY = dotenv.get_key(ENV_FILE, "WANDB_API_KEY")
HF_TOKEN = dotenv.get_key(ENV_FILE, "HF_TOKEN")

# WandB Settings
WANDB_PROJECT = dotenv.get_key(ENV_FILE, "WANDB_PROJECT")
WANDB_RUN_NAME_PREFIX = dotenv.get_key(ENV_FILE, "WANDB_RUN_NAME_PREFIX")

# Hugging Face Settings
HF_USERNAME = dotenv.get_key(ENV_FILE, "HF_USERNAME")
HF_MODEL_PREFIX = dotenv.get_key(ENV_FILE, "HF_MODEL_PREFIX")
HF_MODEL_VERSION = int(dotenv.get_key(ENV_FILE, "HF_MODEL_VERSION"))

# OpenRouter Settings
OPENROUTER_API_URL = dotenv.get_key(ENV_FILE, "OPENROUTER_API_URL")
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}