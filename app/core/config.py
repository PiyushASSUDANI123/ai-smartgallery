import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Facial Recognition Backend"
    VERSION: str = "4.0.0"

    # Paths (Resolving absolute paths relative to this file's location to maintain consistency)
    BACKEND_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PARENT_DIR: str = os.path.dirname(BACKEND_DIR)
    
    # Load from environment variables (.env) or fallback to defaults
# Cloudinary handles all file storage now
    # Supabase / Postgres
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "postgresql://postgres.lpuaxvmqeaijyiqjbpfe:Piyushassudani123@aws-1-ap-south-1.pooler.supabase.com:6543/postgres")

    # Telegram Config
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "8521921505:AAG2XsYipSMFZRhJA10rFd9Cgtu2WGM4jb8")
    TELEGRAM_CHANNEL_ID: str = os.environ.get("TELEGRAM_CHANNEL_ID", "-1003783865322")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Local directories not needed, pipeline is fully in Cloudinary
