import cv2
from deepface import DeepFace
import os

# Directory where the dataset is stored
dataset_dir = "dataset/"

# Ensure that the dataset is present
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
    exit()

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'))

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to test face recognition
def recognize_face(frame):
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop the face from the frame
            face = frame[y:y + h, x:x + w]

            # Resize the face to 224x224 as required by the model
            face_resized = cv2.resize(face, (224, 224))

            # Use DeepFace to recognize the face
            try:
                # The model_name should match the backend used during training
                result = DeepFace.find(face_resized, db_path=dataset_dir, model_name="Facenet", enforce_detection=False)
                print("Recognition Result:", result)
                
                # Display name on the frame
                label = str(result[0]["identity"])
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            except Exception as e:
                print(f"Error during recognition: {e}")
                label = "Unknown"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the recognized face in a separate window
            cv2.imshow("Recognized Face", face_resized)
    
    return frame

# Main loop for testing
print("Testing the face recognition system. Press 'q' to exit.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Recognize faces in the frame
    frame_with_recognition = recognize_face(frame)

    # Display the video feed with recognition result
    cv2.imshow('Video', frame_with_recognition)

    # Check for key press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
