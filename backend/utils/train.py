import os
from deepface import DeepFace
from mtcnn import MTCNN
import cv2
import numpy as np
import pickle

# Directory where the dataset is stored
dataset_dir = "dataset/"

# Ensure that the dataset is present
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
    exit()

# List all the folders (each person) in the dataset directory
people = os.listdir(dataset_dir)

# Check if there are any subdirectories (people)
if not people:
    print("Error: No subdirectories found in the dataset. Make sure you have collected images.")
    exit()

# Set the backend for DeepFace (you can change to VGG-Face, FaceNet, etc.)
backend = "Facenet"  # You can change this to other models like VGG-Face, OpenFace, etc.

# Initialize MTCNN detector for face detection
detector = MTCNN()

# Dictionary to store embeddings for each person
embeddings_dict = {}

# Loop through each person's folder in the dataset directory
for person in people:
    person_dir = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_dir):
        continue

    # List to store face embeddings for the current person
    person_embeddings = []

    # Loop through each image in the person's folder
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        
        # Read the image
        img = cv2.imread(img_path)
        
        # Detect faces using MTCNN
        faces = detector.detect_faces(img)
        
        # If faces are detected, extract embeddings for each face
        if faces:
            for face in faces:
                # Get bounding box coordinates
                x, y, w, h = face['box']
                
                # Crop the face from the image
                face_image = img[y:y+h, x:x+w]
                
                # Use DeepFace to extract embeddings for the face
                embedding = DeepFace.represent(face_image, model_name=backend, enforce_detection=False)
                
                # Append the embedding to the person's list of embeddings
                person_embeddings.append(embedding[0]['embedding'])

    # Store the person's embeddings in the dictionary
    embeddings_dict[person] = person_embeddings

# Save the embeddings to a pickle file for future use
with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)

print("Face embeddings have been successfully extracted and saved.")
