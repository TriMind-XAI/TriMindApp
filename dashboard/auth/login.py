import streamlit as st
from .authDB import init_db, register_user, verify_user
from .googleAuth import get_auth_url, fetch_google_user

init_db()

def login_page():
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image("dashboard/components/assets/logo.png", width=400)
    
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Google"])

    with tab1:
        username = st.text_input("Username",key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state["user"] = username
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("Choose username", key="register_username")
        new_pass = st.text_input("Choose password", type="password", key="register_password")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Account created! Go to Login tab.")
            else:
                st.error("Username already taken")

    with tab3:
        if st.button("Continue with Google", key="google_login"):
            auth_url = get_auth_url()
            st.markdown(f'<meta http-equiv="refresh" content="0;url={auth_url}">', unsafe_allow_html=True)


# Handle Google OAuth callback

def handle_google_callback():
    params = st.query_params

    if "code" in params:
        user_info = fetch_google_user(params["code"])

        st.session_state["user"] = user_info["email"]

        # clear URL so it doesn't repeat
        st.query_params.clear()

        # force rerun into authenticated state
        st.rerun()
