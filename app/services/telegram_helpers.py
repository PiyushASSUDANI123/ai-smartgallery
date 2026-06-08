import requests
import logging
from fastapi import HTTPException
from app.core.config import settings

logger = logging.getLogger("TelegramHelpers")

def get_telegram_file_url(file_id: str) -> str:
    """
    Fetches the temporary file path from Telegram API using the file_id,
    and returns the direct download URL.
    """
    bot_token = settings.TELEGRAM_BOT_TOKEN
    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is missing")
        raise HTTPException(status_code=500, detail="Telegram configuration is missing.")
        
    if file_id.startswith("http://") or file_id.startswith("https://"):
        logger.info(f"Legacy Cloudinary URL detected, skipping Telegram API: {file_id}")
        return file_id
        
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
        response = requests.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok"):
            logger.error(f"Failed to get file from Telegram: {data}")
            raise HTTPException(status_code=500, detail="Failed to fetch file from Telegram.")
            
        file_path = data["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
        return download_url
    except Exception as e:
        logger.error(f"Error fetching file from Telegram: {e}")
        raise HTTPException(status_code=500, detail="Error fetching file from Telegram.")
