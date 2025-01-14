# app/routers/registration.py
from fastapi import APIRouter, File, UploadFile, HTTPException
import os
from deepface import DeepFace
import cv2
import numpy as np
import pickle
from pathlib import Path
import shutil
from model.user import RegistrationRequest, RegistrationResponse, ProcessingResponse
router = APIRouter()

# Configuration
DATASET_DIR = Path("dataset")
EMBEDDINGS_FILE = "face_embeddings.pkl"
BACKEND = "Facenet"

# Ensure dataset directory exists
DATASET_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/init", response_model=RegistrationResponse)
async def initialize_registration(registration: RegistrationRequest):
    """Initialize the registration process for a new employee"""
    try:
        # Create employee directory
        employee_dir = DATASET_DIR / registration.name
        employee_dir.mkdir(parents=True, exist_ok=True)
        
        return RegistrationResponse(
            status="success",
            message=f"Registration initialized for {registration.name}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=RegistrationResponse)
async def upload_face_image(
    name: str,
    position: str,
    index: int,
    file: UploadFile = File(...)
):
    """Upload a single face image during registration"""
    try:
        employee_dir = DATASET_DIR / name
        if not employee_dir.exists():
            raise HTTPException(status_code=404, detail="Employee registration not initialized")
        
        # Save the uploaded image
        image_path = employee_dir / f"{name}_{position}_{index}.jpg"
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return RegistrationResponse(
            status="success",
            message=f"Image uploaded successfully: {image_path.name}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process", response_model=ProcessingResponse)
async def process_registration(name: str):
    """Process all uploaded images and generate embeddings"""
    try:
        person_dir = DATASET_DIR / name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail="Employee data not found")
        
        person_embeddings = []
        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                try:
                    embeddings = DeepFace.represent(img, model_name=BACKEND, enforce_detection=False)
                    for embedding in embeddings:
                        person_embeddings.append(embedding["embedding"])
                except Exception as e:
                    print(f"Error processing {img_file}: {str(e)}")
        
        if not person_embeddings:
            raise HTTPException(status_code=500, detail="No valid embeddings generated")
        
        # Calculate and save average embedding
        aggregated_embedding = np.mean(person_embeddings, axis=0).tolist()
        
        # Load existing embeddings or create new dict
        embeddings_dict = {}
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                embeddings_dict = pickle.load(f)
        
        # Update embeddings
        embeddings_dict[name] = aggregated_embedding
        
        # Save updated embeddings
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings_dict, f)
        
        return ProcessingResponse(
            status="success",
            embedding_path=EMBEDDINGS_FILE
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{name}")
async def check_registration_status(name: str):
    """Check the registration status for an employee"""
    try:
        employee_dir = DATASET_DIR / name
        if not employee_dir.exists():
            return {"status": "not_found"}
        
        # Count images
        image_count = len(list(employee_dir.glob("*.jpg")))
        
        # Check if embeddings exist
        has_embeddings = False
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "rb") as f:
                embeddings_dict = pickle.load(f)
                has_embeddings = name in embeddings_dict
        
        return {
            "status": "complete" if has_embeddings else "in_progress",
            "images_collected": image_count,
            "has_embeddings": has_embeddings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))