import streamlit as st
from streamlit.components.v1 import html
import cv2
import os
from pathlib import Path
from deepface import DeepFace

def nav_to(url):
    js = f"""
        <script>
            window.history.pushState('', '', '{url}');
        </script>
    """
    html(js, height=0)

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
    BASE_DATASET_DIR = Path("datasets")  # You can customize this base directory
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

    positions = ['front', 'left', 'right']

    # Create dataset directory for the user
    user_dir = BASE_DATASET_DIR / st.session_state.name
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating dataset directory: {e}")
        return

    # Initialize face cascade
    face_cascade_path = os.path.join(
        os.path.dirname(cv2.__file__),
        'data',
        'haarcascade_frontalface_default.xml'
    )
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Camera capture section
    instruction = st.empty()
    FRAME_WINDOW = st.image([])

    wait_message = st.markdown(
        "<div style='text-align: center; font-size: 16px; color: gray;'>"
        "Please wait until the webcam starts processing...</div>",
        unsafe_allow_html=True,
    )

    camera = cv2.VideoCapture(0)
    while st.session_state.captured_images < 50:
        instruction.markdown(
            f"<div style='text-align: center; font-size: 20px;'>"
            f"Please look <span style='font-style: italic;'>"
            f"{positions[st.session_state.current_position]}</span></div>",
            unsafe_allow_html=True,
        )

        ret, frame = camera.read()
        if ret:
            wait_message.empty()
            # Convert to RGB for display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    # Add padding to the face
                    padding = 80
                    x = max(x - padding, 0)
                    y = max(y - padding, 0)
                    w = min(w + 2 * padding, frame.shape[1] - x)
                    h = min(h + 2 * padding, frame.shape[0] - y)

                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Extract and save face
                    face = frame[y:y + h, x:x + w]
                    face_resized = cv2.resize(face, (224, 224))

                    # Save image
                    img_path = user_dir / f"{st.session_state.name}_{positions[st.session_state.current_position]}_{st.session_state.captured_images}.jpg"
                    try:
                        cv2.imwrite(str(img_path), face_resized)
                        st.session_state.captured_images += 1
                    except Exception as e:
                        st.error(f"Error saving image: {e}")
                        camera.release()
                        return

                    # Update position every 17 images
                    if st.session_state.captured_images % 17 == 0:
                        st.session_state.current_position = (st.session_state.current_position + 1) % len(positions)

            FRAME_WINDOW.image(display_frame)

            # Check if we're done
            if st.session_state.captured_images >= 50:
                camera.release()
                nav_to("?page=registered")
                st.query_params["page"] = "registered"
                st.rerun()

    camera.release()
    st.write('Camera processing complete!')

def registered_page():
    st.header("Registration Complete!")
    st.success(f"Employee {st.session_state.name} registered successfully!")
    
    if st.button("Return to Home"):
        for key in ['name', 'email', 'employee_id', 'captured_images', 'current_position']:
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