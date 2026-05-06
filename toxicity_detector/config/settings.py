import os
from dotenv import load_dotenv
from toxicity_detector.config.paths import ROOT_DIR


ENV_FILE = ROOT_DIR / ".env"
load_dotenv(ENV_FILE)

# API Keys and Tokens
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# WandB Settings
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_RUN_NAME_PREFIX = os.getenv("WANDB_RUN_NAME_PREFIX")

# Hugging Face Settings
HF_USERNAME = os.getenv("HF_USERNAME")
HF_MODEL_PREFIX = os.getenv("HF_MODEL_PREFIX")
HF_MODEL_VERSION = int(os.getenv("HF_MODEL_VERSION", "1"))

# OpenRouter Settings
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")
OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}