import os
import cloudinary
import cloudinary.uploader
from fastapi import HTTPException, status
from io import BytesIO
from PIL import Image
import pillow_heif
import logging
import asyncio

from app.core.config import settings

logger = logging.getLogger("StorageService")

# Register HEIC opener for PIL
pillow_heif.register_heif_opener()

# Configure Cloudinary
if settings.CLOUDINARY_CLOUD_NAME and settings.CLOUDINARY_API_KEY and settings.CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True
    )
else:
    logger.warning("Cloudinary credentials are not configured. Uploads will fail.")

def upload_to_cloudinary(content: bytes, filename: str, event_id: str) -> str:
    """Uploads a byte string directly to Cloudinary and returns the secure URL."""
    try:
        # Construct a folder path based on event_id for organization
        folder_path = f"gallery_events/{event_id}"
        
        # Cloudinary automatically detects content type, we just pass the bytes
        response = cloudinary.uploader.upload(
            content,
            folder=folder_path,
            public_id=filename,
            overwrite=True,
            resource_type="image"
        )
        
        # Return the secure public URL
        return response.get("secure_url")
    except Exception as e:
        logger.error(f"Failed to upload to Cloudinary: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image to cloud storage.")

async def delete_from_cloudinary(file_url: str) -> bool:
    """
    Deletes an image from Cloudinary using its secure URL.
    Parses the URL to extract the public_id and destroys the asset asynchronously.
    """
    if not settings.CLOUDINARY_CLOUD_NAME or not settings.CLOUDINARY_API_KEY or not settings.CLOUDINARY_API_SECRET:
        logger.warning("Cloudinary credentials not configured, skipping delete.")
        return False
        
    def sync_delete():
        try:
            # A typical URL looks like:
            # https://res.cloudinary.com/<cloud_name>/image/upload/v<version>/<public_id_path>
            if "image/upload/" not in file_url:
                logger.warning(f"URL is not a standard Cloudinary image URL: {file_url}")
                return False
                
            parts = file_url.split("image/upload/")
            path_part = parts[1] # e.g. v1570975253/gallery_events/event1/pic.jpg
            
            # Remove version if present (starts with 'v' followed by digits)
            path_segments = path_part.split("/")
            if path_segments[0].startswith("v") and path_segments[0][1:].isdigit():
                path_segments = path_segments[1:]
                
            # Reconstruct public_id without extension
            public_id_with_ext = "/".join(path_segments)
            public_id = os.path.splitext(public_id_with_ext)[0]
            
            logger.info(f"Destroying Cloudinary asset with public_id: {public_id}")
            response = cloudinary.uploader.destroy(public_id)
            result = response.get("result")
            logger.info(f"Cloudinary destroy response: {response}")
            return result == "ok"
        except Exception as e:
            logger.error(f"Failed to delete from Cloudinary: {e}")
            return False

    return await asyncio.to_thread(sync_delete)

async def process_and_save_uploaded_image(content: bytes, filename: str, event_id: str) -> tuple[str, int]:
    """
    Validates any uploaded image format. 
    - HEIC/HEIF/TIFF: converts to JPEG at maximum quality (browsers can't display these).
    - JPEG/PNG/WebP/BMP/GIF: uploads the ORIGINAL bytes UNCHANGED — no quality loss.
    Returns (public_url, file_size).
    """
    safe_base = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(filename)[1].lower()
    
    def process_image_sync():
        try:
            image = Image.open(BytesIO(content))
        except Exception as e:
            logger.error(f"Failed to parse image {filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is not a valid image format."
            )

        # Check if HEIC/HEIF or TIFF - browsers cannot display these, must convert
        is_heic = ext in {".heic", ".heif"} or (hasattr(image, 'format') and image.format in {"HEIF", "HEIC"})
        is_tiff = ext in {".tif", ".tiff"} or (hasattr(image, 'format') and image.format == "TIFF")
        
        file_size = len(content)
        upload_content = content
        
        if is_heic or is_tiff:
            # Convert to JPEG at MAXIMUM quality (100) — no lossy compression
            if image.mode != "RGB":
                rgb_im = image.convert("RGB")
            else:
                rgb_im = image
                
            img_byte_arr = BytesIO()
            rgb_im.save(img_byte_arr, format="JPEG", quality=100, subsampling=0)
            upload_content = img_byte_arr.getvalue()
            file_size = len(upload_content)
            
            if rgb_im is not image:
                rgb_im.close()
            image.close()
            logger.info(f"Converted {filename} from {ext} to JPEG-100 ({file_size} bytes)")
        else:
            # JPEG, PNG, WebP, BMP, GIF — upload original bytes as-is, NO compression
            image.close()
            logger.info(f"Uploading {filename} as original ({file_size} bytes, no conversion)")
            
        return upload_content, file_size

    # CPU-bound PIL tasks in background thread
    upload_content, file_size = await asyncio.to_thread(process_image_sync)
        
    # Upload to Cloudinary (Async wrapper)
    public_url = await asyncio.to_thread(upload_to_cloudinary, upload_content, safe_base, event_id)
    logger.info(f"Successfully uploaded {filename} to Cloudinary: {public_url}")
    
    return public_url, file_size


