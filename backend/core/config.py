import os
from pathlib import Path
from dotenv import load_dotenv

# Find the project root directory where .env is located
env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

SETTINGS = Settings()
