import dotenv
from toxicity_detector.config.paths import ROOT_DIR

ENV_FILE = ROOT_DIR / ".env"
dotenv.load_dotenv(ENV_FILE)

# API Keys
OPENROUTER_API_KEY = dotenv.get_key(ENV_FILE, "OPENROUTER_API_KEY")

# API Endpoints
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}