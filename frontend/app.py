import streamlit as st
from streamlit.components.v1 import html
import cv2
import os
from pathlib import Path
from deepface import DeepFace
import requests

def nav_to(url):
    js = f"""
        <script>
            window.history.pushState('', '', '{url}');
        </script>
    """
    html(js, height=0)

def register_api_call(name, employee_id, image_paths):
    """
    Make API call to register a new employee using multipart form data.
    Considers registration successful if at least 40 valid face images are detected.
    """
    API_ENDPOINT = "http://localhost:8000/api/register"
    MIN_VALID_FRAMES = 40
    
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
        invalid_images = []
        
        for i, path in enumerate(image_paths):
            try:
                # Read and verify image
                img = cv2.imread(str(path))
                if img is None:
                    print(f"Warning: Could not read image {path}")
                    invalid_images.append((str(path), "Could not read image"))
                    continue
                    
                # Verify image dimensions
                height, width = img.shape[:2]
                if height < 64 or width < 64:
                    print(f"Warning: Image {path} is too small ({width}x{height})")
                    invalid_images.append((str(path), "Image too small"))
                    continue
                
                # Verify face detection using DeepFace.represent
                try:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face_result = DeepFace.represent(
                        rgb_img, 
                        model_name="Facenet",
                        enforce_detection=True,
                        detector_backend="opencv"
                    )
                    
                    if face_result and isinstance(face_result, list) and len(face_result) > 0:
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        files.append(
                            ("images", (f"face_{i}.jpg", buffer.tobytes(), "image/jpeg"))
                        )
                        valid_images.append(str(path))
                        print(f"Successfully processed image: {path}")
                    else:
                        print(f"Warning: No face detected in {path}")
                        invalid_images.append((str(path), "No face detected"))
                        continue
                    
                except Exception as e:
                    print(f"Warning: Face detection failed for {path}: {str(e)}")
                    invalid_images.append((str(path), f"Face detection failed: {str(e)}"))
                    continue
                    
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                invalid_images.append((str(path), f"Processing error: {str(e)}"))
                continue
        
        valid_count = len(valid_images)
        print(f"Processed {valid_count} valid images out of {len(image_paths)}")
        
        if valid_count < MIN_VALID_FRAMES:
            return {
                "success": False,
                "message": f"Insufficient valid face images. Found {valid_count}, need at least {MIN_VALID_FRAMES}",
                "status_code": 400,
                "valid_images": valid_images,
                "invalid_images": invalid_images,
                "total_processed": valid_count
            }
        
        print(f"Sending {valid_count} valid images to API")
        
        # Make the API call with multipart form data
        response = requests.post(
            API_ENDPOINT,
            files=files,
            data={
                "name": name,
                "employee_id": employee_id
            },
            timeout=30
        )
        
        print(f"API Response: {response.status_code} - {response.text}")
        
        return {
            "success": response.status_code == 200,
            "message": response.text,
            "status_code": response.status_code,
            "valid_images": valid_images,
            "invalid_images": invalid_images,
            "total_processed": valid_count
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

def register_page1():
    st.header("Step 1: Basic Information")
    st.write("Fill in your details below to register.")

    # Initialize session state if not already set
    if 'name' not in st.session_state:
        st.session_state.name = ""
    if 'employee_id' not in st.session_state:
        st.session_state.employee_id = ""

    with st.form("register_form"):
        # Inputs for Name and Employee ID
        name = st.text_input("Full Name", value=st.session_state.name)
        employee_id = st.text_input("Employee ID", value=st.session_state.employee_id)
        submitted = st.form_submit_button("Next")
        
        if submitted:
            # Validate inputs
            if name and employee_id:
                # Store in session state and navigate to next step
                st.session_state.name = name
                st.session_state.employee_id = employee_id
                # Navigate to Step 2 (register_page2)
                nav_to("?page=register2")
                st.query_params["page"] = "register2"
                st.rerun()  # Rerun the app to reflect changes
            else:
                st.error("Please fill out all the fields.")

# def register_page1():
#     st.header("Step 1: Basic Information")
#     st.write("Fill in your details below to register.")
    
#     # Default values for skipping the page
#     default_name = "John Doe"
#     default_employee_id = "12345"
    
#     # Use default values if not already in session state
#     if "name" not in st.session_state:
#         st.session_state.name = default_name
#     if "employee_id" not in st.session_state:
#         st.session_state.employee_id = default_employee_id
    
#     with st.form("register_form"):
#         # Prepopulate fields with default values
#         name = st.text_input("Full Name", value=st.session_state.name)
#         employee_id = st.text_input("Employee ID", value=st.session_state.employee_id)
#         submitted = st.form_submit_button("Next")
        
#         if submitted:
#             if name and employee_id:
#                 st.session_state.name = name
#                 st.session_state.employee_id = employee_id
#                 nav_to("?page=register2")
#                 st.query_params["page"] = "register2"
#                 st.rerun()
#             else:
#                 st.error("Please fill out all the fields.")

def register_page2():
    BASE_DATASET_DIR = Path("datasets")
    if not hasattr(st.session_state, 'name'):
        st.error("Please complete step 1 first.")
        nav_to("?page=register1")
        st.query_params["page"] = "register1"
        st.rerun()
        return

    st.header("Step 2: Face Registration")

    # Initialize capture variables in session state
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = 0
        st.session_state.current_position = 0
        st.session_state.last_capture_time = 0

    positions = ['front', 'left', 'right']

    # Create dataset directory for the user
    user_dir = BASE_DATASET_DIR / st.session_state.name
    user_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face cascade with optimized parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Camera capture section
    instruction = st.empty()
    FRAME_WINDOW = st.image([])
    progress_bar = st.progress(0)
    status_text = st.empty()

    camera = cv2.VideoCapture(0)
    # Set lower resolution for faster processing
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    import time
    capture_interval = 0.1  # Minimum time between captures in seconds

    try:
        while st.session_state.captured_images < 50:
            current_time = time.time()
            instruction.markdown(
                f"<div style='text-align: center; font-size: 20px;'>"
                f"Please look <span style='font-style: italic;'>"
                f"{positions[st.session_state.current_position]}</span></div>",
                unsafe_allow_html=True,
            )

            ret, frame = camera.read()
            if not ret:
                continue

            # Convert to grayscale first for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optimize face detection parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Increased for faster processing
                minNeighbors=3,   # Reduced for faster detection
                minSize=(50, 50)  # Minimum face size
            )

            if len(faces) > 0 and (current_time - st.session_state.last_capture_time) >= capture_interval:
                x, y, w, h = faces[0]  # Take only the first face detected
                
                # Add padding to the face
                padding = 50  # Reduced padding for faster processing
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = min(w + 2 * padding, frame.shape[1] - x)
                h = min(h + 2 * padding, frame.shape[0] - y)

                # Extract and save face
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (160, 160))  # Smaller size for faster processing

                # Save image
                img_path = user_dir / f"{st.session_state.name}_{positions[st.session_state.current_position]}_{st.session_state.captured_images}.jpg"
                cv2.imwrite(str(img_path), face_resized)
                
                st.session_state.captured_images += 1
                st.session_state.last_capture_time = current_time

                # Update position counter
                if st.session_state.captured_images % 17 == 0:
                    st.session_state.current_position = (st.session_state.current_position + 1) % len(positions)

            # Update UI
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            FRAME_WINDOW.image(display_frame)
            progress_bar.progress(st.session_state.captured_images / 50)
            status_text.text(f"Captured {st.session_state.captured_images}/50 images")

            if st.session_state.captured_images >= 50:
                camera.release()
                nav_to("?page=registered")
                st.query_params["page"] = "registered"
                st.rerun()

    finally:
        camera.release()    

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