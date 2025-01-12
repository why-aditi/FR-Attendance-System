import cv2
from deepface import DeepFace
import os

dataset_dir = "dataset/"
if not os.path.exists(dataset_dir):
    print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
    exit()

face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'))

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def recognize_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (224, 224))
            try:
                result = DeepFace.find(face_resized, db_path=dataset_dir, model_name="Facenet", enforce_detection=False)
                if result:
                    full_identity_path = result[0].iloc[0]["identity"]
                    name = os.path.basename(os.path.dirname(full_identity_path))
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error during recognition: {e}")
                cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

print("Testing the face recognition system. Press 'q' to exit.")
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_with_recognition = recognize_face(frame)

    cv2.imshow('Video', frame_with_recognition)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
