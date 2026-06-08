import os
import cloudinary
import cloudinary.uploader
from fastapi import HTTPException, status
from io import BytesIO
from PIL import Image
import pillow_heif
import logging

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
    """Uploads a byte string directly to Cloudinary and returns the public URL."""
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

def process_and_save_uploaded_image(content: bytes, filename: str, event_id: str) -> tuple[str, int]:
    """
    Validates any uploaded image format. If HEIC/HEIF or TIFF,
    converts it to JPEG with high quality (95) on the fly without loss.
    Uploads the final image to Cloudinary and returns (public_url, file_size).
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
        
    # Upload to Cloudinary
    public_url = upload_to_cloudinary(upload_content, safe_base, event_id)
    logger.info(f"Successfully uploaded {filename} to Cloudinary: {public_url}")
    
    return public_url, file_size
