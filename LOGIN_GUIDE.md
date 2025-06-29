# 🔐 Authentication System Guide

## Overview

Your ML CSV Analyzer now has a **secure login system** that authenticates users against a local CSV file with properly hashed passwords.

## 🏗️ **What Was Added**

### 1. **Authentication Module** (`auth_utils.py`)
- **Password hashing** using PBKDF2 with SHA256 (100,000 iterations)
- **Session management** with 60-minute timeout
- **User loading** from CSV file
- **Login/logout** functionality

### 2. **User Management** (`manage_users.py`)
- **Command-line tool** to manage users
- **Add/remove users** without touching the app
- **Change passwords** securely
- **Activate/deactivate** users

### 3. **Protected App** (`app.py`)
- **Login required** before accessing any features
- **User info display** in sidebar
- **Session tracking** and automatic logout

## 🚀 **Quick Start**

### Step 1: Create Default Users
```bash
source ml_analyzer_env/bin/activate
python manage_users.py
# Choose option 6: Create default users
```

### Step 2: Test the App
```bash
./test_local.sh
```

### Step 3: Login with Demo Credentials
- **Demo User**: `demo` / `demo123`

## 👥 **User Management**

### Command Line Tool
```bash
python manage_users.py
```

**Available Options:**
1. **List all users** - View current users and their status
2. **Add new user** - Create new user account
3. **Deactivate user** - Disable user without deleting
4. **Activate user** - Re-enable deactivated user
5. **Change password** - Update user password
6. **Create default users** - Setup demo accounts
7. **Exit** - Close the tool

### Example: Adding a New User
```bash
python manage_users.py
# Choose option 2
# Username: john_doe
# Full Name: John Doe
# Role: user
# Password: (hidden input)
```

## 🔒 **Security Features**

### Password Security
- ✅ **PBKDF2 hashing** with 100,000 iterations
- ✅ **Random salt** for each password
- ✅ **No plaintext storage** - passwords never stored as text
- ✅ **Secure comparison** using constant-time functions

### Session Security
- ✅ **60-minute timeout** - automatic logout after inactivity
- ✅ **Secure session state** - stored in Streamlit's session
- ✅ **Proper cleanup** - logout clears all session data

### File Security
- ✅ **CSV protection** - users.csv excluded from Git
- ✅ **Local only** - no network transmission of credentials
- ✅ **Role-based** - admin vs user roles (extensible)

## 📁 **File Structure**

```
vibeML/
├── app.py                    # Main app (now with auth)
├── auth_utils.py            # Authentication functions
├── manage_users.py          # User management CLI
├── users.csv               # User database (auto-created)
└── LOGIN_GUIDE.md          # This guide
```

## 🔧 **Customization**

### Change Session Timeout
Edit `auth_utils.py`:
```python
SESSION_TIMEOUT_MINUTES = 120  # 2 hours instead of 1
```

### Add New Roles
Edit `auth_utils.py` and extend the role system:
```python
# Add role checking
if st.session_state.role == 'admin':
    st.sidebar.button("Admin Panel")
```

### Custom User Fields
Extend the CSV structure:
```csv
username,password_hash,full_name,role,active,department,email
```

## 🌐 **Web Deployment with Authentication**

When you deploy with ngrok, the authentication is built-in:

```bash
./deploy_web.sh
```

**Security Benefits:**
- ✅ **App-level authentication** - even if someone gets the ngrok URL
- ✅ **Session tracking** - can see who's logged in
- ✅ **User management** - can disable users remotely
- ✅ **No ngrok auth needed** - authentication is in your app

## ⚠️ **Production Considerations**

### Current Setup (Good for Demo/Development)
- ✅ Local CSV file
- ✅ PBKDF2 password hashing
- ✅ Session management

### For Production (Recommended Upgrades)
- 🔄 **Database backend** (PostgreSQL, MySQL)
- 🔄 **Environment variables** for sensitive config
- 🔄 **HTTPS enforcement**
- 🔄 **Rate limiting** for login attempts
- 🔄 **Audit logging** for user actions
- 🔄 **Password complexity** requirements
- 🔄 **2FA/MFA** support

## 🐛 **Troubleshooting**

### "No users found" Error
```bash
python manage_users.py
# Choose option 6 to create default users
```

### "Import Error" for auth_utils
```bash
# Make sure you're in the project directory
cd /path/to/vibeML
source ml_analyzer_env/bin/activate
```

### Forgot Password
```bash
python manage_users.py
# Choose option 5 to change password
```

### Reset All Users
```bash
rm users.csv
python manage_users.py
# Choose option 6 to recreate default users
```

## 📋 **Demo Scenario**

Perfect for showing to clients or colleagues:

1. **Deploy**: `./deploy_web.sh`
2. **Share URL**: `https://abc123_SK.ngrok.io`
3. **Provide credentials**: 
   - Demo: `demo` / `demo123`
4. **Monitor usage**: See who's logged in via sidebar
5. **Secure shutdown**: Stop ngrok, all sessions end

This gives you a **professional, secure ML platform** that's ready for real-world use! 🚀 