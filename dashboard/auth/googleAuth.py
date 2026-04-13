# google_auth.py
import streamlit as st
from authlib.integrations.httpx_client import OAuth2Client

def get_google_client():
    return OAuth2Client(
        client_id=st.secrets["google_oauth"]["client_id"],
        client_secret=st.secrets["google_oauth"]["client_secret"],
    )

def get_auth_url():
    client = get_google_client()
    uri, state = client.create_authorization_url(
        "https://accounts.google.com/o/oauth2/auth",
        redirect_uri=st.secrets["google_oauth"]["redirect_uri"],
        scope="openid email profile",
    )
    st.session_state["oauth_state"] = state
    return uri

def fetch_google_user(code):
    client = get_google_client()
    token = client.fetch_token(
        "https://oauth2.googleapis.com/token",
        code=code,
        redirect_uri=st.secrets["google_oauth"]["redirect_uri"],
    )
    userinfo = client.get("https://www.googleapis.com/oauth2/v3/userinfo").json()
    return userinfo  # has .email, .name, .sub