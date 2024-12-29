#Load libraries needed
import streamlit as st


# Set page configuration 
st.set_page_config(
    page_title="Norbel AI Platform",
    page_icon="https://github.com/EnidVyshka/Norbel_AI/blob/master/logo.png?raw=true",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Function to display the common navigation buttons at the top
def display_navigation_buttons():
    # Buttons for navigation
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏚️ Home", use_container_width=True):
            st.session_state.page = "Home"

    with col3:
        if st.button("About Us", use_container_width=True):
            st.session_state.page = "Page 1"

    with col2:
        if st.button("Sales Prediction", use_container_width=True):
            st.session_state.page = "Page 2"


# Home Page
def home():
    st.markdown(" # 👋 Welcome to the Norbel AI")
    st.image("logo.PNG")


# Page 1
def page1():
    from pages.About import about_page
    about_page()


# Page 2
def page2():
    from pages.Sales_Pred import sales_prediction_page
    sales_prediction_page()
    # st.title("Page 2")
    # st.write("This is Page 2.")


# Main function to manage pages
def main():
    # Initialize session state for page navigation if not already initialized
    if "page" not in st.session_state:
        st.session_state.page = "Home"  # Default starting page

    # Display navigation buttons
    display_navigation_buttons()

    # Navigate to the page based on session state
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Page 1":
        page1()
    elif st.session_state.page == "Page 2":
        page2()


# Run the app
if __name__ == "__main__":
    main()
