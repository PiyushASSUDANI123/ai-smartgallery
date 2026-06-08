import os
import requests
from fastapi import HTTPException, status
from io import BytesIO
from PIL import Image
import pillow_heif
import logging

from app.core.config import settings

logger = logging.getLogger("StorageService")

# Register HEIC opener for PIL
pillow_heif.register_heif_opener()

def upload_to_telegram(content: bytes, filename: str, event_id: str) -> str:
    """Uploads a byte string to Telegram Private Channel and returns the file_id."""
    bot_token = settings.TELEGRAM_BOT_TOKEN
    channel_id = settings.TELEGRAM_CHANNEL_ID
    
    if not bot_token or not channel_id:
        logger.error("Telegram credentials are not configured.")
        raise HTTPException(status_code=500, detail="Telegram configuration is missing.")
        
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
        
        # Send as document to prevent compression and keep original quality
        files = {
            "document": (filename, content, "image/jpeg")
        }
        data = {
            "chat_id": channel_id,
            "caption": f"Event ID: {event_id} | File: {filename}"
        }
        
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        
        resp_data = response.json()
        if not resp_data.get("ok"):
            logger.error(f"Telegram upload failed: {resp_data}")
            raise HTTPException(status_code=500, detail="Failed to upload image to Telegram.")
            
        # Extract file_id from document
        document = resp_data["result"].get("document")
        if document:
            return document["file_id"]
        
        # Fallback if sent as photo
        photo = resp_data["result"].get("photo")
        if photo:
            return photo[-1]["file_id"]  # Last one is the highest resolution
            
        raise HTTPException(status_code=500, detail="Invalid response from Telegram.")
            
    except Exception as e:
        logger.error(f"Failed to upload to Telegram: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image to cloud storage.")

def process_and_save_uploaded_image(content: bytes, filename: str, event_id: str) -> tuple[str, int]:
    """
    Validates any uploaded image format. If HEIC/HEIF or TIFF,
    converts it to JPEG with high quality (95) on the fly without loss.
    Uploads the final image to Telegram and returns (file_id, file_size).
    """
    safe_base = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        image = Image.open(BytesIO(content))
    except Exception as e:
        logger.error(f"Failed to parse image {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is not a valid image format."
        )

    # Check if HEIC/HEIF or TIFF
    is_heic = ext in {".heic", ".heif"} or (hasattr(image, 'format') and image.format in {"HEIF", "HEIC"})
    is_tiff = ext in {".tif", ".tiff"} or (hasattr(image, 'format') and image.format == "TIFF")
    
    file_size = len(content)
    upload_content = content
    final_ext = ext
    
    if is_heic or is_tiff:
        final_ext = ".jpg"
        
        # Convert to RGB (JPEG doesn't support transparency/RGBA)
        if image.mode != "RGB":
            rgb_im = image.convert("RGB")
        else:
            rgb_im = image
            
        img_byte_arr = BytesIO()
        rgb_im.save(img_byte_arr, format="JPEG", quality=95)
        upload_content = img_byte_arr.getvalue()
        file_size = len(upload_content)
        
        if rgb_im is not image:
            rgb_im.close()
        image.close()
        logger.info(f"Converted {filename} to JPEG ({file_size} bytes)")
    else:
        image.close()
        
    final_filename = f"{safe_base}{final_ext}"
        
    # Upload to Telegram
    file_id = upload_to_telegram(upload_content, final_filename, event_id)
    logger.info(f"Successfully uploaded {filename} to Telegram with file_id: {file_id}")
    
    return file_id, file_size
