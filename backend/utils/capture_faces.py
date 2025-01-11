import cv2
import os

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory to store collected images
dataset_dir = "dataset/"

# Create directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Get the person's name and create a subdirectory for them
name = input("Enter your name: ")
person_dir = os.path.join(dataset_dir, name)

if not os.path.exists(person_dir):
    os.makedirs(person_dir)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize image count and position labels
img_count = 0
positions = ['front', 'left', 'right']
current_position = 0

print("Collecting images. Look at the camera in different positions (front, left, right).")

# Loop to capture images
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Ensure a face is detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Add some padding around the face (less tight crop)
            padding = 80  # Increase or decrease this value to adjust the crop
            x = max(x - padding, 0)
            y = max(y - padding, 0)
            w = min(w + 2 * padding, frame.shape[1] - x)  # Prevent going out of bounds
            h = min(h + 2 * padding, frame.shape[0] - y)

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the face region with padding
            face = frame[y:y + h, x:x + w]

            # Save the image with the current position label
            img_count += 1
            img_path = os.path.join(person_dir, f"{name}_{positions[current_position]}_{img_count}.jpg")
            cv2.imwrite(img_path, face)

            # Show the captured face in a separate window
            cv2.imshow('Captured Face', face)

    # Show the video feed
    cv2.imshow('Video', frame)

    # Stop if 100 images have been collected
    if img_count >= 100:
        break

    # Check for key press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # After capturing images for a specific position, switch to the next
    if img_count % 34 == 0 and img_count != 0:
        current_position = (current_position + 1) % 3
        print(f"Change position to {positions[current_position]}")

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

print(f"Dataset for {name} collected successfully!")
