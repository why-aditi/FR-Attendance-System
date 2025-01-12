import cv2
import os
import time

face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'))

dataset_dir = "dataset/"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

name = input("Enter your name: ")
person_dir = os.path.join(dataset_dir, name)

if not os.path.exists(person_dir):
    os.makedirs(person_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_count = 0
positions = ['front', 'left', 'right']
current_position = 0

print("Collecting images. Look at the camera in different positions (front, left, right).")

capture_interval = 0.2  

while img_count < 50:  
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            padding = 80  
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, frame.shape[1] - x)
            h = min(h + 2 * padding, frame.shape[0] - y)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (224, 224))
            img_count += 1
            img_path = os.path.join(person_dir, f"{name}_{positions[current_position]}_{img_count}.jpg")
            cv2.imwrite(img_path, face_resized)
            cv2.imshow('Captured Face', face_resized)
            time.sleep(capture_interval)
            if img_count >= 50:
                break

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if img_count % 17 == 0 and img_count != 0:
        current_position = (current_position + 1) % len(positions)
        print(f"Change position to {positions[current_position]}")

cap.release()
cv2.destroyAllWindows()

print(f"Dataset for {name} collected successfully!")
