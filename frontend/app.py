import streamlit as st
from streamlit.components.v1 import html

def nav_to(url):
    # JavaScript to handle navigation without opening new tab
    js = f"""
        <script>
            window.history.pushState('', '', '{url}');
        </script>
    """
    html(js, height=0)

def main():
    # Set up the title and description
    st.set_page_config(page_title="Employee Clock-In System", page_icon="📅")
    
    # Get the current page from URL parameters
    current_page = st.query_params.get("page", "home")

    # Display appropriate page based on URL
    if current_page == "register":
        register_page()
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

    # Buttons for navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register 👤"):
            nav_to("?page=register")
            st.query_params["page"] = "register"
            st.rerun()

    with col2:
        if st.button("Clock In ⏰"):
            nav_to("?page=clock-in")
            st.query_params["page"] = "clock-in"
            st.rerun()

def register_page():
    st.title("Employee Clock-In System")
    st.header("Employee Registration")
    st.write("Fill in your details below to register.")

    # Back button
    if st.button("← Back to Home"):
        nav_to("/")
        st.query_params.clear()
        st.rerun()

    with st.form("register_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        employee_id = st.text_input("Employee ID")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if name and email and employee_id:
                st.success(f"Employee {name} registered successfully!")
            else:
                st.error("Please fill out all the fields.")

def clock_in_page():
    st.title("Employee Clock-In System")
    st.header("Employee Clock-In")
    st.write("Enter your Employee ID to clock in.")

    # Back button
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