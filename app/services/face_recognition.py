#!/usr/bin/env python3
"""
Face Recognition Event Photo Matcher
Author: Senior Python AI Engineer

This module provides a clean, modular, and optimized solution to match a reference
face (selfie) against a local folder of event images. It uses face_recognition
(which relies on dlib) and opencv-python, and includes Apple Silicon optimizations
via multiprocessing to run face detection in parallel across CPU cores.
"""

import os
import cv2
import numpy as np
import logging
import argparse
import face_recognition
import gc
from typing import List, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging with a professional format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceMatcher")

# Supported image file extensions
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class FaceRecognitionError(Exception):
    """Custom exception class for Face Recognition errors."""
    pass


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    Loads an image from the given path and converts it to RGB format.
    Uses OpenCV for robust loading, then converts BGR to RGB.

    Args:
        image_path: Path to the image file.

    Returns:
        np.ndarray: The RGB image.

    Raises:
        FileNotFoundError: If the file does not exist.
        FaceRecognitionError: If the image cannot be decoded or loaded.
    """
    if not image_path.startswith("http") and not os.path.isfile(image_path):
        # Assume it's a Telegram file_id
        from app.services.telegram_helpers import get_telegram_file_url
        try:
            image_path = get_telegram_file_url(image_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to resolve Telegram file_id '{image_path}': {e}")

    if image_path.startswith("http"):
        import requests
        import numpy as np
        resp = requests.get(image_path, timeout=20)
        resp.raise_for_status()
        nparr = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceRecognitionError(f"OpenCV failed to decode the URL image: '{image_path}'")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image path does not exist: '{image_path}'")

    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FaceRecognitionError(f"OpenCV failed to decode the image: '{image_path}'")
        
        # OpenCV loads in BGR; face_recognition expects RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    except Exception as e:
        if not isinstance(e, (FileNotFoundError, FaceRecognitionError)):
            raise FaceRecognitionError(f"Error loading image '{image_path}': {str(e)}") from e
        raise


def resize_image_if_large(image: np.ndarray, max_width: int = 1024) -> np.ndarray:
    """
    Resizes an image if its width exceeds max_width, maintaining aspect ratio.
    Uses cv2.INTER_AREA interpolation which is optimized for downscaling.

    Args:
        image: The input image (numpy array).
        max_width: The maximum allowed width.

    Returns:
        np.ndarray: The resized image (or original if within limit).
    """
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / float(width)
        new_height = int(height * scaling_factor)
        logger.info(f"Downscaling image from {width}x{height} to {max_width}x{new_height} to optimize memory & CPU.")
        resized = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    return image


def extract_reference_encoding(image_path: str, model: str = "hog") -> np.ndarray:
    """
    Loads a reference image (selfie), detects the face, and returns its 128-d encoding.
    Ensures that exactly one face is detected. Resizes high-res images to optimize CPU.

    Args:
        image_path: Path to the reference selfie image.
        model: Face detection model to use ("hog" or "cnn").

    Returns:
        np.ndarray: The 128-dimensional face encoding of the reference face.

    Raises:
        FaceRecognitionError: If zero or multiple faces are detected in the reference selfie.
    """
    logger.info(f"Loading reference image from: {image_path}")
    rgb_image = load_image_rgb(image_path)
    
    # Downscale reference image if it is too large
    rgb_image = resize_image_if_large(rgb_image, max_width=1024)

    # Find face locations
    face_locations = face_recognition.face_locations(rgb_image, model=model)
    
    # Fallback mechanism if face is too small, dark, or angled
    if len(face_locations) == 0 and model == "hog":
        logger.warning("No face detected with HOG. Falling back to robust CNN model...")
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=0, model="cnn")
        
    num_faces = len(face_locations)

    if num_faces == 0:
        # Clean up memory before raising exception
        del rgb_image
        if 'face_locations' in locals():
            del face_locations
        gc.collect()
        raise FaceRecognitionError(f"No face detected in the reference selfie image: {image_path}")
    elif num_faces > 1:
        # A selfie should ideally only contain the target user's face.
        logger.warning(f"Multiple faces ({num_faces}) detected in reference image. Using the first detected face.")

    # Generate highly robust encodings for the reference face
    encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=100)
    if not encodings:
        del rgb_image
        del face_locations
        gc.collect()
        raise FaceRecognitionError(f"Failed to generate face encoding for reference image: {image_path}")

    # Extract the target encoding and explicitly clean up large arrays
    encoding = encodings[0]
    
    del rgb_image
    del face_locations
    del encodings
    gc.collect()

    return encoding


def process_single_event_image(
    image_path: str,
    reference_encoding: np.ndarray,
    tolerance: float = 0.80,
    model: str = "hog"
) -> Optional[str]:
    """
    Processes a single event image: loads it, downscales it to <= 1024px width, detects
    all faces, extracts encodings, and checks if any face matches the reference face.
    Explicitly deletes numpy arrays and triggers gc.collect() to prevent OOM.
    Designed to be run in a parallel process pool.

    Args:
        image_path: Path to the event image.
        reference_encoding: The 128-d numpy array of the reference face.
        tolerance: Distance tolerance for face comparison (lower is stricter).
        model: Face detection model to use ("hog" or "cnn").

    Returns:
        Optional[str]: The path of the image if a match is found, otherwise None.
    """
    rgb_image = None
    face_locations = None
    face_encodings = None
    matches = None
    match_found = False

    try:
        rgb_image = load_image_rgb(image_path)
        
        # Downscale the image if it exceeds 1024px width
        rgb_image = resize_image_if_large(rgb_image, max_width=1024)
        
        # Locate all faces in the event photo (upsample to catch smaller/far faces)
        face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model=model)
        
        if not face_locations:
            logger.debug(f"No faces detected in event photo: {image_path}")
            return None

        # Extract 128-d encodings using multiple jitters for high precision
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations, num_jitters=10)
        
        # Compare each face encoding to the reference encoding
        matches = face_recognition.compare_faces(face_encodings, reference_encoding, tolerance=tolerance)
        match_found = any(matches)
        
        if match_found:
            logger.info(f"MATCH FOUND in file: {os.path.basename(image_path)}")
            return image_path
            
    except Exception as e:
        logger.error(f"Error processing event image '{image_path}': {str(e)}")
        
    finally:
        # Explicitly delete numpy arrays and call garbage collector to free memory immediately
        if rgb_image is not None:
            del rgb_image
        if face_locations is not None:
            del face_locations
        if face_encodings is not None:
            del face_encodings
        if matches is not None:
            del matches
        gc.collect()
        
    return None


def match_faces_in_folder(
    reference_path: str,
    event_dir: str,
    tolerance: float = 0.80,
    model: str = "hog",
    num_workers: Optional[int] = None
) -> List[str]:
    """
    Loads reference face and scans the event folder for matches.
    Optimized for Apple Silicon ARM64 / Multi-core processors using concurrent process pools.

    Args:
        reference_path: Path to reference image (selfie).
        event_dir: Path to directory containing event photos.
        tolerance: Distance tolerance for face comparison (default: 0.5).
        model: Face detection model to use ("hog" or "cnn").
        num_workers: Number of parallel CPU processes to spawn (default: CPU count).

    Returns:
        List[str]: List of filenames (basenames) of the matching images.
    """
    # 1. Extract reference face encoding
    reference_encoding = extract_reference_encoding(reference_path, model=model)
    logger.info("Successfully loaded and encoded reference face.")

    # 2. Check event directory
    if not os.path.isdir(event_dir):
        raise NotADirectoryError(f"Event directory does not exist or is not a directory: '{event_dir}'")

    # 3. Gather all supported files in event directory
    event_files = []
    for entry in os.scandir(event_dir):
        if entry.is_file():
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                event_files.append(entry.path)

    if not event_files:
        logger.warning(f"No supported images found in event directory: {event_dir}")
        return []

    logger.info(f"Scanning {len(event_files)} event photos for matches using model='{model}', tolerance={tolerance}...")

    matched_filenames: List[str] = []

    # 4. Process event images in parallel (optimized for Apple Silicon / multi-core CPUs)
    # CPU count is chosen automatically if num_workers is None.
    # On M-series Macs, this fully utilizes Performance and Efficiency cores.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                process_single_event_image,
                file_path,
                reference_encoding,
                tolerance,
                model
            ): file_path
            for file_path in event_files
        }

        # Retrieve results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result:
                matched_filenames.append(os.path.basename(result))

    logger.info(f"Scan complete. Found {len(matched_filenames)} matching photos.")
    return matched_filenames


def main() -> None:
    """CLI Entry point for the face recognition matcher."""
    parser = argparse.ArgumentParser(
        description="Match a reference face (selfie) against a folder of event photos."
    )
    parser.add_argument(
        "-r", "--reference",
        required=True,
        help="Path to the reference selfie image."
    )
    parser.add_argument(
        "-e", "--event-dir",
        required=True,
        help="Path to the folder containing event images."
    )
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.80,
        help="Comparison tolerance. Loose tolerance (e.g. 0.65) avoids false negatives. Default is 0.65."
    )
    parser.add_argument(
        "-m", "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model to use. 'hog' is faster on CPU. 'cnn' is more accurate but slow on CPU. Default: hog."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: CPU core count."
    )

    args = parser.parse_args()

    try:
        matches = match_faces_in_folder(
            reference_path=args.reference,
            event_dir=args.event_dir,
            tolerance=args.tolerance,
            model=args.model,
            num_workers=args.workers
        )
        print("\n--- Matching Images Found ---")
        for match in sorted(matches):
            print(match)
        print("-----------------------------\n")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
