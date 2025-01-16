from fastapi import UploadFile, File, HTTPException, APIRouter, Form
from deepface import DeepFace
import numpy as np
import cv2
import json
from config.db import face_collection

router = APIRouter()

# Utility functions
def get_face_embedding(image: np.ndarray) -> np.ndarray:
    """Extract face embedding using DeepFace."""
    try:
        # Generate face embedding
        embedding = DeepFace.represent(
            image,
            model_name="Facenet",
            enforce_detection=True
        )
        return np.array(embedding)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract face embedding: {str(e)}"
        )

def process_image(image_data: bytes) -> np.ndarray:
    """Convert image bytes to a numpy array and validate the image."""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data. Unable to decode.")
    return img

# Route handlers
@router.post("/register", response_model=dict)
async def register_employee(name: str = Form(...),
    employee_id: str = Form(...),
    images: list[UploadFile] = File(...)):
    """
    Register a new employee with multiple face images.

    Args:
        employee (NewEmployee): Employee details (name and employee_id).
        images (List[UploadFile]): List of face image files uploaded.

    Returns:
        dict: Registration status and employee details.
    """
    try:
        # Check if employee already exists in the database
        existing = face_collection.get(where={"employee_id": employee_id})
        if existing["ids"]:
            raise HTTPException(
                status_code=400,
                detail=f"Employee with ID '{employee_id}' already exists."
            )

        # Initialize list for face embeddings
        embeddings = []
        for image in images:
            # Read and process each image
            contents = await image.read()
            img = process_image(contents)

            # Extract face embedding
            embedding = get_face_embedding(img)
            embeddings.append(embedding.tolist())

        # Store embeddings and metadata in the database
        face_collection.add(
            embeddings=embeddings,
            documents=[
                json.dumps({
                    "name": name,
                    "employee_id": employee_id
                })
            ] * len(embeddings),
            ids=[f"{employee_id}_{i}" for i in range(len(embeddings))]
        )

        # Return success response
        return {
            "message": "Employee registered successfully.",
            "employee": {
                "name": name,
                "employee_id": employee_id,
                "faces_registered": len(embeddings)
            }
        }
    except HTTPException as he:
        # Re-raise HTTP exceptions for appropriate response codes
        raise he
    except Exception as e:
        # Handle unexpected exceptions
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
