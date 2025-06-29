import streamlit as st
import pandas as pd
import hashlib
import hmac
import csv
import os
from datetime import datetime, timedelta

# Configuration
USERS_CSV_PATH = "users.csv"
SESSION_TIMEOUT_MINUTES = 60

def hash_password(password: str, salt: str = None) -> tuple:
    """
    Hash a password using PBKDF2 with SHA256
    Returns (hash, salt)
    """
    if salt is None:
        salt = os.urandom(16).hex()
    
    # Use PBKDF2 for secure password hashing
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    ).hex()
    
    return f"pbkdf2_sha256$100000${salt}${password_hash}", salt

def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored hash
    """
    try:
        method, iterations, salt, hash_value = stored_hash.split('$')
        
        # Recreate the hash with the provided password
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            int(iterations)
        ).hex()
        
        return hmac.compare_digest(hash_value, password_hash)
    except Exception:
        return False

def load_users() -> pd.DataFrame:
    """
    Load users from CSV file
    """
    try:
        if os.path.exists(USERS_CSV_PATH):
            return pd.read_csv(USERS_CSV_PATH)
        else:
            # Create default users file if it doesn't exist
            create_default_users()
            return pd.read_csv(USERS_CSV_PATH)
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return pd.DataFrame()

def create_default_users():
    """
    Create a default users.csv file with sample users
    """
    default_users = [
        {
            'username': 'admin',
            'password_hash': hash_password('admin123')[0],
            'full_name': 'Administrator',
            'role': 'admin',
            'active': True
        },
        {
            'username': 'demo',
            'password_hash': hash_password('demo123')[0],
            'full_name': 'Demo User',
            'role': 'user',
            'active': True
        },
        {
            'username': 'analyst',
            'password_hash': hash_password('analyst123')[0],
            'full_name': 'Data Analyst',
            'role': 'user',
            'active': True
        }
    ]
    
    df = pd.DataFrame(default_users)
    df.to_csv(USERS_CSV_PATH, index=False)

def authenticate_user(username: str, password: str) -> dict:
    """
    Authenticate user credentials
    Returns user data if successful, None if failed
    """
    users_df = load_users()
    
    if users_df.empty:
        return None
    
    # Find user
    user_row = users_df[users_df['username'] == username]
    
    if user_row.empty:
        return None
    
    user = user_row.iloc[0]
    
    # Check if user is active
    if not user.get('active', True):
        return None
    
    # Verify password
    if verify_password(password, user['password_hash']):
        return {
            'username': user['username'],
            'full_name': user['full_name'],
            'role': user['role'],
            'login_time': datetime.now()
        }
    
    return None

def is_session_valid() -> bool:
    """
    Check if current session is valid (user logged in and not expired)
    """
    if 'authenticated' not in st.session_state:
        return False
    
    if not st.session_state.authenticated:
        return False
    
    # Check session timeout
    if 'login_time' in st.session_state:
        login_time = st.session_state.login_time
        if datetime.now() - login_time > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            logout_user()
            return False
    
    return True

def login_user(user_data: dict):
    """
    Set session state for logged in user
    """
    st.session_state.authenticated = True
    st.session_state.username = user_data['username']
    st.session_state.full_name = user_data['full_name']
    st.session_state.role = user_data['role']
    st.session_state.login_time = user_data['login_time']

def logout_user():
    """
    Clear session state (logout)
    """
    for key in ['authenticated', 'username', 'full_name', 'role', 'login_time']:
        if key in st.session_state:
            del st.session_state[key]

def require_auth(func):
    """
    Decorator to require authentication for a function
    """
    def wrapper(*args, **kwargs):
        if not is_session_valid():
            show_login_page()
            return None
        return func(*args, **kwargs)
    return wrapper

def show_login_page():
    """
    Display the login page
    """
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>🔐 ML CSV Analyzer</h1>
            <h3>Please log in to continue</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Create login form
    with st.form("login_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Login")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    user_data = authenticate_user(username, password)
                    
                    if user_data:
                        login_user(user_data)
                        st.success(f"Welcome, {user_data['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    # Demo credentials info
    with st.expander("📋 Demo Credentials", expanded=False):
        st.markdown("""
        **Demo Account:**
        - **Demo User**: `demo` / `demo123`  
        
        ⚠️ **Note**: Change these credentials in production!
        """)

def show_user_info():
    """
    Show logged in user info in sidebar
    """
    if is_session_valid():
        st.sidebar.markdown(f"**👤 Welcome, {st.session_state.full_name}**")
        st.sidebar.markdown(f"Role: {st.session_state.role}")
        
        # Session info
        if 'login_time' in st.session_state:
            time_logged_in = datetime.now() - st.session_state.login_time
            st.sidebar.markdown(f"Logged in: {time_logged_in.seconds // 60} min ago")
        
        # Logout button
        if st.sidebar.button("🚪 Logout"):
            logout_user()
            st.rerun()
        
        st.sidebar.markdown("---")

def add_user(username: str, password: str, full_name: str, role: str = "user") -> bool:
    """
    Add a new user to the CSV file
    """
    try:
        users_df = load_users()
        
        # Check if username already exists
        if username in users_df['username'].values:
            return False
        
        # Create new user
        password_hash, _ = hash_password(password)
        new_user = {
            'username': username,
            'password_hash': password_hash,
            'full_name': full_name,
            'role': role,
            'active': True
        }
        
        # Add to dataframe and save
        new_row = pd.DataFrame([new_user])
        users_df = pd.concat([users_df, new_row], ignore_index=True)
        users_df.to_csv(USERS_CSV_PATH, index=False)
        
        return True
    except Exception:
        return False 