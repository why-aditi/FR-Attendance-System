import streamlit as st
from streamlit.components.v1 import html
import cv2
import os
from pathlib import Path

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
    st.title("Employee Clock-In System")
    st.header("Step 1: Basic Information")
    st.write("Fill in your details below to register.")

    if st.button("← Back to Home"):
        nav_to("/")
        st.query_params.clear()
        st.rerun()

    with st.form("register_form"):
        name = st.text_input("Full Name")
        employee_id = st.text_input("Employee ID")
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
        st.error("Please complete step 1 first")
        nav_to("?page=register1")
        st.query_params["page"] = "register1"
        st.rerun()
        return

    st.title("Employee Clock-In System")
    st.header("Step 2: Face Registration")
    st.write("Please look at the camera and follow the instructions.")

    if st.button("← Back to Step 1"):
        nav_to("?page=register1")
        st.query_params["page"] = "register1"
        st.rerun()

    # Initialize capture variables in session state
    if 'captured_images' not in st.session_state:
        st.session_state.captured_images = 0
        st.session_state.current_position = 0

    positions = ['front', 'left', 'right']
    
    # Show current position instruction
    st.info(f"Please look {positions[st.session_state.current_position]}")
    st.write(f"Captured Images: {st.session_state.captured_images}/50")

    # Create dataset directory
    dataset_dir = Path("../dataset") / st.session_state.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(
        os.path.join(os.path.dirname(cv2.__file__), 
        'data', 'haarcascade_frontalface_default.xml')
    )

    # Camera capture section
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    if run:
        camera = cv2.VideoCapture(0)
        while run and st.session_state.captured_images < 50:
            ret, frame = camera.read()
            if ret:
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
                        img_path = dataset_dir / f"{st.session_state.name}_{positions[st.session_state.current_position]}_{st.session_state.captured_images}.jpg"
                        cv2.imwrite(str(img_path), face_resized)
                        
                        st.session_state.captured_images += 1
                        
                        # Update position every 17 images
                        if st.session_state.captured_images % 17 == 0:
                            st.session_state.current_position = (st.session_state.current_position + 1) % len(positions)
                            st.info(f"Please look {positions[st.session_state.current_position]}")
                
                FRAME_WINDOW.image(display_frame)
                
                # Check if we're done
                if st.session_state.captured_images >= 50:
                    camera.release()
                    nav_to("?page=registered")
                    st.query_params["page"] = "registered"
                    st.rerun()
        
        if not run:
            camera.release()
    else:
        st.write('Camera Stopped')

def registered_page():
    st.title("Employee Clock-In System")
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
    st.title("Employee Clock-In System")
    st.header("Employee Clock-In")
    st.write("Enter your Employee ID to clock in.")

    if st.button("← Back to Home"):
        nav_to("/")
        st.query_params.clear()
        st.rerun()

    with st.form("clock_in_form"):
        employee_id = st.text_input("Employee ID")
        submitted = st.form_submit_button("Clock In")

        if submitted:
            if employee_id:
                st.success(f"Clock-In successful for Employee ID: {employee_id}")
            else:
                st.error("Please enter your Employee ID.")

if __name__ == "__main__":
    main()