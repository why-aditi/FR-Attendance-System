import os
from deepface import DeepFace
import cv2
import numpy as np
import pickle
from tqdm import tqdm

# Update the dataset directory path to match your project structure
dataset_dir = "./datasets/"

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Error: Dataset directory '{dataset_dir}' does not exist.")

people = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]

if not people:
    raise ValueError("Error: No subdirectories found in the dataset. Make sure you have collected images.")

backend = "Facenet"
embeddings_dict = {}

for person in tqdm(people, desc="Processing People"):
    person_dir = os.path.join(dataset_dir, person)
    person_embeddings = []
    
    for img_file in tqdm(os.listdir(person_dir), desc=f"Processing Images for {person}", leave=False):
        img_path = os.path.join(person_dir, img_file)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read image {img_path}. Skipping.")
            continue
        
        try:
            embeddings = DeepFace.represent(img, model_name=backend, enforce_detection=False)
            for embedding in embeddings:
                person_embeddings.append(embedding["embedding"])
        
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}. Skipping.")
    
    if person_embeddings:
        aggregated_embedding = np.mean(person_embeddings, axis=0).tolist()
        embeddings_dict[person] = aggregated_embedding
    else:
        print(f"Warning: No embeddings found for {person}. Skipping.")

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)

print("Face embeddings have been successfully extracted and saved.")
