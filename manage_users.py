#!/usr/bin/env python3
"""
User Management Script for ML CSV Analyzer
Allows adding, removing, and managing users from command line
"""

import sys
import getpass
from auth_utils import (load_users, add_user, hash_password, create_default_users)
import pandas as pd

def show_menu():
    """Display the main menu"""
    print("\n🔐 ML CSV Analyzer - User Management")
    print("=" * 40)
    print("1. List all users")
    print("2. Add new user")
    print("3. Deactivate user")
    print("4. Activate user") 
    print("5. Change user password")
    print("6. Create default users")
    print("7. Exit")
    print("-" * 40)

def list_users():
    """List all users"""
    users_df = load_users()
    if users_df.empty:
        print("❌ No users found. Create default users first.")
        return
    
    print("\n👥 Current Users:")
    print("-" * 60)
    for _, user in users_df.iterrows():
        status = "✅ Active" if user.get('active', True) else "❌ Inactive"
        print(f"Username: {user['username']:<15} | Name: {user['full_name']:<20} | Role: {user['role']:<10} | {status}")

def add_new_user():
    """Add a new user"""
    print("\n➕ Add New User")
    print("-" * 20)
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    full_name = input("Full Name: ").strip()
    if not full_name:
        print("❌ Full name cannot be empty")
        return
    
    role = input("Role (admin/user) [user]: ").strip().lower() or "user"
    if role not in ['admin', 'user']:
        print("❌ Role must be 'admin' or 'user'")
        return
    
    password = getpass.getpass("Password: ")
    if len(password) < 6:
        print("❌ Password must be at least 6 characters")
        return
    
    password_confirm = getpass.getpass("Confirm Password: ")
    if password != password_confirm:
        print("❌ Passwords don't match")
        return
    
    if add_user(username, password, full_name, role):
        print(f"✅ User '{username}' added successfully!")
    else:
        print(f"❌ Failed to add user '{username}'. Username might already exist.")

def change_user_status(activate=True):
    """Activate or deactivate a user"""
    users_df = load_users()
    if users_df.empty:
        print("❌ No users found.")
        return
    
    action = "activate" if activate else "deactivate"
    print(f"\n{'🟢' if activate else '🔴'} {action.title()} User")
    print("-" * 20)
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    # Find user
    user_idx = users_df[users_df['username'] == username].index
    if user_idx.empty:
        print(f"❌ User '{username}' not found")
        return
    
    # Update status
    users_df.loc[user_idx, 'active'] = activate
    users_df.to_csv('users.csv', index=False)
    
    print(f"✅ User '{username}' {'activated' if activate else 'deactivated'} successfully!")

def change_password():
    """Change user password"""
    users_df = load_users()
    if users_df.empty:
        print("❌ No users found.")
        return
    
    print("\n🔑 Change Password")
    print("-" * 20)
    
    username = input("Username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    # Find user
    user_idx = users_df[users_df['username'] == username].index
    if user_idx.empty:
        print(f"❌ User '{username}' not found")
        return
    
    password = getpass.getpass("New Password: ")
    if len(password) < 6:
        print("❌ Password must be at least 6 characters")
        return
    
    password_confirm = getpass.getpass("Confirm New Password: ")
    if password != password_confirm:
        print("❌ Passwords don't match")
        return
    
    # Hash new password
    password_hash, _ = hash_password(password)
    
    # Update password
    users_df.loc[user_idx, 'password_hash'] = password_hash
    users_df.to_csv('users.csv', index=False)
    
    print(f"✅ Password for '{username}' changed successfully!")

def create_defaults():
    """Create default users"""
    print("\n🔧 Creating Default Users")
    print("-" * 25)
    print("This will create default users including:")
    print("- demo / demo123 (Demo User)")
    print("- Additional system users for administration")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm == 'y':
        create_default_users()
        print("✅ Default users created successfully!")
    else:
        print("❌ Operation cancelled")

def main():
    """Main program loop"""
    while True:
        show_menu()
        choice = input("Select option (1-7): ").strip()
        
        try:
            if choice == '1':
                list_users()
            elif choice == '2':
                add_new_user()
            elif choice == '3':
                change_user_status(activate=False)
            elif choice == '4':
                change_user_status(activate=True)
            elif choice == '5':
                change_password()
            elif choice == '6':
                create_defaults()
            elif choice == '7':
                print("👋 Goodbye!")
                sys.exit(0)
            else:
                print("❌ Invalid option. Please choose 1-7.")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 