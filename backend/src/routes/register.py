from fastapi import UploadFile, File, HTTPException, APIRouter, Form, Request
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import cv2
import json
import logging
from config.db import face_collection
from typing import List, Optional
import io
from PIL import Image
from bson import Binary
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/jpg'}
MIN_IMAGE_SIZE = 64
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

def log_image_info(image: UploadFile, prefix: str = ""):
    """Log image metadata for debugging."""
    logger.info(f"{prefix} Image Details:")
    logger.info(f"Filename: {image.filename}")
    logger.info(f"Content-Type: {image.content_type}")
    try:
        logger.info(f"File size: {image.size} bytes")
    except:
        logger.info("File size: Unknown")

async def validate_image_file(image: UploadFile) -> bool:
    """
    Validate uploaded image file before processing.
    Returns True if valid, raises HTTPException if invalid.
    """
    if not image:
        raise HTTPException(
            status_code=400,
            detail="No image file provided"
        )
    
    log_image_info(image, "Validating")
    
    if not image.content_type in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {image.filename}. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )
    
    try:
        contents = await image.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Empty file: {image.filename}"
            )
        
        if len(contents) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {image.filename}. Maximum size: {MAX_IMAGE_SIZE/1024/1024}MB"
            )
        
        # Try to open the image to verify it's valid
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
            width, height = img.size
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image too small: {width}x{height}. Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {image.filename}. Error: {str(e)}"
            )
        
        # Reset file pointer
        await image.seek(0)
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating {image.filename}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Error validating {image.filename}: {str(e)}"
        )

def get_face_embedding(image: np.ndarray) -> np.ndarray:
    """Extract face embedding with multiple detection attempts."""
    try:
        logger.info("Starting face embedding extraction")
        logger.info(f"Input image shape: {image.shape}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # List of detection attempts with different preprocessing
        detection_attempts = [
            # 1. Original image
            (image_rgb, "original"),
            
            # 2. Enhanced contrast
            (cv2.convertScaleAbs(image_rgb, alpha=1.5, beta=0), "contrast enhanced"),
            
            # 3. Histogram equalization
            (cv2.cvtColor(
                cv2.equalizeHist(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)),
                cv2.COLOR_GRAY2RGB
            ), "histogram equalized"),
            
            # 4. Resized image if too large
            (cv2.resize(image_rgb, (640, 480)) if image_rgb.shape[0] > 640 or image_rgb.shape[1] > 480 
             else image_rgb, "resized")
        ]
        
        last_error = None
        for img, attempt_name in detection_attempts:
            try:
                logger.info(f"Attempting face detection with {attempt_name} image")
                results = DeepFace.represent(
                    img,
                    model_name="Facenet",
                    enforce_detection=True,
                    detector_backend="opencv"  # Try using OpenCV detector
                )
                
                if results and isinstance(results, list) and 'embedding' in results[0]:
                    logger.info(f"Face detection successful with {attempt_name} image")
                    return np.array(results[0]["embedding"])
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Face detection failed with {attempt_name} image: {last_error}")
                continue
        
        # If all attempts failed, try one last time with enforce_detection=False
        try:
            logger.info("Attempting face detection with enforce_detection=False")
            results = DeepFace.represent(
                image_rgb,
                model_name="Facenet",
                enforce_detection=False
            )
            
            if results and isinstance(results, list) and 'embedding' in results[0]:
                logger.info("Face detection successful with enforce_detection=False")
                return np.array(results[0]["embedding"])
                
        except Exception as e:
            last_error = str(e)
            logger.error(f"Final face detection attempt failed: {last_error}")
        
        raise ValueError(f"Face detection failed after all attempts. Last error: {last_error}")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Face embedding extraction failed: {error_message}")
        raise HTTPException(
            status_code=400,
            detail="No face detected. Please ensure the image contains a clear, well-lit face looking directly at the camera. Last error: " + error_message
        )


def process_image(image_data: bytes) -> np.ndarray:
    """Process image data into numpy array."""
    try:
        logger.info("Processing image data")
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image data")
            
        logger.info(f"Image processed successfully. Shape: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )

@router.post("/register")
async def register_employee(
    request: Request,
    name: str = Form(...),
    employee_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Register an employee with their face embeddings using ChromaDB.
    """
    try:
        logger.info(f"Starting registration for employee_id: {employee_id}")
        
        # Validate input
        if not name or not employee_id:
            raise HTTPException(
                status_code=400,
                detail="Name and employee_id are required"
            )
        
        if not images or len(images) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one image is required"
            )
        
        logger.info(f"Received {len(images)} images")
        
        # Process each image
        embeddings = []
        metadatas = []
        failed_images = []
        for idx, image in enumerate(images):
            try:
                logger.info(f"Processing image {idx + 1}/{len(images)}: {image.filename}")
                
                # Validate image
                await validate_image_file(image)
                
                # Read image data
                image_data = await image.read()
                
                # Process image
                img = process_image(image_data)
                
                # Extract face embedding
                embedding = get_face_embedding(img)
                embeddings.append(embedding.tolist())  # Convert numpy array to list for ChromaDB
                
                # Add metadata for the embedding
                metadatas.append({
                    "filename": image.filename,
                    "employee_id": employee_id,
                    "name": name
                })
                
                logger.info(f"Successfully processed image {idx + 1}")
                
            except HTTPException as e:
                logger.error(f"Failed to process image {idx + 1}: {str(e)}")
                failed_images.append(image.filename)
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing image {idx + 1}: {str(e)}")
                failed_images.append(image.filename)
                continue
        
        # Store embeddings in ChromaDB
                # Store embeddings in ChromaDB
        if embeddings:
            try:
                logger.info("Storing embeddings in ChromaDB")
                
                # Print embeddings in the backend console
                logger.info(f"Embeddings: {embeddings}")
                
                if embeddings:
                    print(f"First embedding: {embeddings[0]}")  # This will print the first embedding
                    l = len(embeddings)
                else:
                    print("No embeddings found.")
                # Ensure face_collection is a valid ChromaDB collection
                if not hasattr(face_collection, "add"):
                    raise HTTPException(
                        status_code=500,
                        detail="ChromaDB collection is not properly initialized"
                    )
                
                # Add embeddings and metadata to ChromaDB
                face_collection.add(
                    embeddings=embeddings,  # List of embeddings
                    metadatas=metadatas,   # List of metadata dictionaries
                    ids=[f"{employee_id}_{i}" for i in range(len(embeddings))]  # Unique IDs for each embedding
                )
                
                logger.info(f"Successfully stored embeddings for employee_id: {employee_id}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": "Registration successful",
                        "employee_id": employee_id,
                        "faces_registered": l,
                        "failed_images": failed_images
                    }
                )
            except Exception as e:
                logger.error(f"Failed to store embeddings in ChromaDB: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to store embeddings in ChromaDB: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="No valid face embeddings were extracted from the provided images"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during registration: {str(e)}"
        )      
        
        