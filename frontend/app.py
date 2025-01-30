import cv2
import numpy as np
from pathlib import Path
import streamlit as st
import time
import json
import streamlit as st
from streamlit.components.v1 import html
import cv2
import os
from pathlib import Path
from deepface import DeepFace
import requests
import numpy as np
from deepface import DeepFace  # Import DeepFace for embedding generation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs

def nav_to(url):
    js = f"""
        <script>
            window.history.pushState('', '', '{url}');
        </script>
    """
    html(js, height=0)

def register_api_call(name, employee_id, image_paths):
    """
    Make API call to register a new employee using multipart form data with improved error handling
    and image validation using correct DeepFace API
    """
    API_ENDPOINT = "http://localhost:8000/api/register"
    
    try:
        # Validate inputs
        if not name or not employee_id:
            return {
                "success": False,
                "message": "Name and employee ID are required",
                "status_code": 400
            }
            
        if not image_paths:
            return {
                "success": False,
                "message": "No image paths provided",
                "status_code": 400
            }
        
        # Prepare files list for multipart upload
        files = []
        valid_images = []
        
        for i, path in enumerate(image_paths):
            try:
                # Read and verify image
                img = cv2.imread(str(path))
                if img is None:
                    print(f"Warning: Could not read image {path}")
                    continue
                    
                # Verify image dimensions
                height, width = img.shape[:2]
                if height < 64 or width < 64:
                    print(f"Warning: Image {path} is too small ({width}x{height})")
                    continue
                
                # Verify face detection using DeepFace.represent
                try:
                    # Convert BGR to RGB for DeepFace
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Use represent function with enforce_detection=True to ensure face is present
                    face_result = DeepFace.represent(
                        rgb_img, 
                        model_name="Facenet",
                        enforce_detection=True,
                        detector_backend="opencv"
                    )
                    
                    if face_result and isinstance(face_result, list) and len(face_result) > 0:
                        # If face detection succeeds, add to valid images
                        # Encode as JPEG with quality setting
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        files.append(
                            ("images", (f"face_{i}.jpg", buffer.tobytes(), "image/jpeg"))
                        )
                        valid_images.append(path)
                        print(f"Successfully processed image: {path}")
                    else:
                        print(f"Warning: No face detected in {path}")
                        continue
                    
                except Exception as e:
                    print(f"Warning: Face detection failed for {path}: {str(e)}")
                    continue
                    
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                continue
        
        if not files:
            return {
                "success": False,
                "message": "No valid face images found in the provided images",
                "status_code": 400
            }
        
        print(f"Sending {len(files)} valid images to API")
        
        # Make the API call with multipart form data
        response = requests.post(
            API_ENDPOINT,
            files=files,
            data={
                "name": name,
                "employee_id": employee_id
            },
            timeout=30  # Add timeout
        )
        
        print(f"API Response: {response.status_code} - {response.text}")
        
        return {
            "success": response.status_code == 200,
            "message": response.text,
            "status_code": response.status_code,
            "valid_images": valid_images,
            "total_processed": len(valid_images)
        }
        
    except requests.exceptions.RequestException as e:
        print(f"API Request Error: {str(e)}")
        return {
            "success": False,
            "message": f"API request failed: {str(e)}",
            "status_code": None
        }
    finally:
        # Clean up
        for file_tuple in files:
            try:
                if hasattr(file_tuple[1][1], 'close'):
                    file_tuple[1][1].close()
            except:
                pass
            
def main():
    st.set_page_config(page_title="Employee Clock-In System", page_icon="📅")
    
    current_page = st.query_params.get("page", "home")

    if current_page == "register1":
        register_page1()
    elif current_page == "register2":
        register_page2()
    elif current_page == "registered":
        registered_page()
    elif current_page == "clock-in":
        clock_in_page()
    else:
        home_page()

def home_page():
    st.title("Employee Clock-In System")
    st.write("\n")
    st.markdown(
        """
        ### Choose an Option Below:
        1. *Register*: For new employees to register in the system.
        2. *Clock In*: For existing employees to clock in.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register 👤"):
            nav_to("?page=register1")
            st.query_params["page"] = "register1"
            st.rerun()

    with col2:
        if st.button("Clock In ⏰"):
            nav_to("?page=clock-in")
            st.query_params["page"] = "clock-in"
            st.rerun()

# def register_page1():
#     st.header("Step 1: Basic Information")
#     st.write("Fill in your details below to register.")

#     # Initialize session state if not already set
#     if 'name' not in st.session_state:
#         st.session_state.name = ""
#     if 'employee_id' not in st.session_state:
#         st.session_state.employee_id = ""

#     with st.form("register_form"):
#         # Inputs for Name and Employee ID
#         name = st.text_input("Full Name", value=st.session_state.name)
#         employee_id = st.text_input("Employee ID", value=st.session_state.employee_id)
#         submitted = st.form_submit_button("Next")
        
#         if submitted:
#             # Validate inputs
#             if name and employee_id:
#                 # Store in session state and navigate to next step
#                 st.session_state.name = name
#                 st.session_state.employee_id = employee_id
#                 # Navigate to Step 2 (register_page2)
#                 nav_to("?page=register2")
#                 st.query_params["page"] = "register2"
#                 st.rerun()  # Rerun the app to reflect changes
#             else:
#                 st.error("Please fill out all the fields.")

def register_page1():
    st.header("Step 1: Basic Information")
    st.write("Fill in your details below to register.")
    
    # Default values for skipping the page
    default_name = "John Doe"
    default_employee_id = "12345"
    
    # Use default values if not already in session state
    if "name" not in st.session_state:
        st.session_state.name = default_name
    if "employee_id" not in st.session_state:
        st.session_state.employee_id = default_employee_id
    
    with st.form("register_form"):
        # Prepopulate fields with default values
        name = st.text_input("Full Name", value=st.session_state.name)
        employee_id = st.text_input("Employee ID", value=st.session_state.employee_id)
        submitted = st.form_submit_button("Next")
        
        if submitted:
            if name and employee_id:
                st.session_state.name = name
                st.session_state.employee_id = employee_id
                nav_to("?page=register2")
                st.query_params["page"] = "register2"
                st.rerun()
            else:
                st.error("Please fill out all the fields.")

def register_page2():
    if not hasattr(st.session_state, 'name'):
        st.error("Please complete step 1 first.")
        nav_to("?page=register1")
        st.query_params["page"] = "register1"
        st.rerun()
        return

    st.header("Step 2: Face Registration")

    # Add debug container
    debug_container = st.empty()

    # Initialize capture variables in session state
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = 0
        st.session_state.current_position = 0
        st.session_state.last_capture_time = 0
        st.session_state.processed_images = []

    TOTAL_REQUIRED_IMAGES = 5
    positions = ['front', 'left', 'right']

    # Camera capture section
    instruction = st.empty()
    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)
    status_text = st.empty()
    api_status = st.empty()

    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def process_face(face_img):
        try:
            if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                return None

            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_result = DeepFace.represent(
                rgb_face,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )

            if face_result and isinstance(face_result, list) and len(face_result) > 0:
                embedding = face_result[0]["embedding"]
                print(f"Face Embedding: {embedding[:5]}... (truncated)")  # Print first 5 values for brevity

                
                return face_img
            return None
            
        except Exception as e:
            debug_container.text(f"Face processing error: {str(e)}")
            return None



    def send_batch_to_api(files):
        try:
            status_text.text("Processing and uploading images...")
            debug_container.text("Sending API request...")

            # Show the payload size for debugging
            total_size = sum(len(f[1][1]) for f in files)
            debug_container.text(f"Total payload size: {total_size / 1024 / 1024:.2f} MB")
            
            # Increased timeout and added connection timeout
            response = requests.post(
                "http://localhost:8000/api/register",
                files=files,
                data={
                    "name": st.session_state.name,
                    "employee_id": st.session_state.employee_id
                },
                timeout=(5, 60)  # (connection timeout, read timeout)
            )
            
            debug_container.text(f"Server response status: {response.status_code}")
            
            try:
                response_json = response.json()
                debug_container.text(f"Server response: {json.dumps(response_json, indent=2)}")
                
                if response.status_code == 200:
                    faces_registered = response_json.get('employee', {}).get('faces_registered', 0)
                    failed = len(response_json.get('failed_images', []))
                    
                    status_text.success(f"Successfully registered {faces_registered} faces. Failed: {failed}")
                    return True
                else:
                    error_detail = response_json.get('detail', 'Unknown error')
                    status_text.error(f"Registration failed: {error_detail}")
                    return False
                    
            except json.JSONDecodeError:
                status_text.error("Failed to parse server response")
                debug_container.text(f"Raw response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            status_text.error("Request timed out. The server is taking too long to process the images.")
            return False
        except requests.exceptions.RequestException as e:
            status_text.error(f"Network error: {str(e)}")
            debug_container.text(f"Request error details: {str(e)}")
            return False



    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    capture_interval = 2.0  # Time in seconds between captures

    try:
        while st.session_state.captured_images < TOTAL_REQUIRED_IMAGES:
            current_time = time.time()
            current_position = st.session_state.current_position % len(positions)
            
            instruction.markdown(
                f"<div style='text-align: center; font-size: 20px;'>"
                f"Please look <span style='font-style: italic;'>"
                f"{positions[current_position]}</span></div>",
                unsafe_allow_html=True,
            )

            ret, frame = camera.read()
            if not ret:
                debug_container.text("Failed to capture frame from camera.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )

            if len(faces) > 0 and (current_time - st.session_state.last_capture_time) >= capture_interval:
                x, y, w, h = faces[0]
                
                padding = 40
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = min(w + 2 * padding, frame.shape[1] - x)
                h = min(h + 2 * padding, frame.shape[0] - y)

                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (160, 160))
                
                print("Face detected, processing...")  # **New debug print**

                processed_face = process_face(face_resized)  
                
                if processed_face is not None:
                    _, buffer = cv2.imencode('.jpg', processed_face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    st.session_state.processed_images.append(
                        ("images", (f"face_{st.session_state.captured_images}.jpg", buffer.tobytes(), "image/jpeg"))
                    )
                    
                    st.session_state.captured_images += 1
                    st.session_state.last_capture_time = current_time

                    if st.session_state.captured_images % 2 == 0:
                        st.session_state.current_position = (st.session_state.current_position + 1) % len(positions)

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            FRAME_WINDOW.image(display_frame)
            progress_bar.progress(st.session_state.captured_images / TOTAL_REQUIRED_IMAGES)
            status_text.text(f"Captured {st.session_state.captured_images}/{TOTAL_REQUIRED_IMAGES} images")

            if st.session_state.captured_images >= TOTAL_REQUIRED_IMAGES:
                if send_batch_to_api(st.session_state.processed_images):
                    time.sleep(2)  # Give time for success message to be seen
                    camera.release()
                    nav_to("?page=registered")
                    st.query_params["page"] = "registered"
                    st.rerun()
                else:
                    # Reset for retry
                    st.session_state.captured_images = 0
                    st.session_state.processed_images = []

    finally:
        camera.release()
        st.session_state.processed_images = []





def registered_page():
    # Check if we have the required session state
    if 'name' not in st.session_state or 'employee_id' not in st.session_state:
        st.error("Session expired or invalid. Please start registration again.")
        if st.button("Return to Registration"):
            nav_to("?page=register1")
            st.query_params["page"] = "register1"
            st.rerun()
        return
    
    st.header("Registration Complete!")
    
    dataset_path = Path("datasets") / st.session_state.name
    if dataset_path.exists():
        image_paths = list(dataset_path.glob("*.jpg"))
        with st.spinner("Syncing with server..."):
            api_response = register_api_call(
                st.session_state.name, 
                st.session_state.employee_id, 
                image_paths
            )
            
            if api_response["success"]:
                st.success(f"Employee {st.session_state.name} registered successfully!")
                st.write(f"Registered {len(api_response.get('valid_images', []))} face images")
            else:
                st.error("Server registration failed")
                st.write(api_response["message"])
                st.write("Please contact support with the following details:")
                st.code(api_response)
    else:
        st.error(f"No images found for {st.session_state.name}")
    
    if st.button("Return to Home"):
        # Clear session state
        for key in ['name', 'employee_id', 'captured_images', 'current_position', 'last_capture_time']:
            if key in st.session_state:
                del st.session_state[key]
        
        nav_to("/")
        st.query_params.clear()
        st.rerun()

def clock_in_page():    
    # Dataset directory for face recognition
    dataset_dir = Path("datasets") 
    if not os.path.exists(dataset_dir):
        st.error(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        st.stop()

    face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'))

    st.header("Employee Clock-In")
    st.write("Use the webcam to verify your identity and clock in.")

    if st.button("← Back to Home"):
        nav_to("/")
        st.query_params.clear()
        st.rerun()

    if st.button("Start Face Recognition"):
        # Start the webcam for face recognition
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return

        st.info("Press 'q' in the window to stop the face recognition test.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to capture frame.")
                break

            # Face detection and recognition
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face = frame[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (224, 224))

                    try:
                        result = DeepFace.find(face_resized, db_path=str(dataset_dir), model_name="Facenet", enforce_detection=False)
                        if result and not result[0].empty:
                            full_identity_path = result[0].iloc[0]["identity"]
                            name = os.path.basename(os.path.dirname(full_identity_path))
                            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Error during recognition: {e}")
                        cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the video feed with recognition results
            cv2.imshow("Face Recognition Test", frame)

            # Press 'q' to exit the video feed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()