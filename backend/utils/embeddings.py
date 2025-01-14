import cv2
import numpy as np
from deepface import DeepFace

def extract_face_embedding(image_path: str, backend: str = "Facenet"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image {image_path}.")
    
    try:
        embeddings = DeepFace.represent(img, model_name=backend, enforce_detection=False)
        if embeddings:
            return np.mean([embedding["embedding"] for embedding in embeddings], axis=0).tolist()
        else:
            raise ValueError(f"No embeddings found for {image_path}.")
    except Exception as e:
        raise ValueError(f"Error extracting embeddings from {image_path}: {str(e)}")