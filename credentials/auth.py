import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

CREDENTIALS_DIR = Path(__file__).parent
USERS_FILE = CREDENTIALS_DIR / 'users.json'
HISTORY_FILE = CREDENTIALS_DIR / 'password_history.json'

def hash_password(password):
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def initialize_credentials():
    """Initialize credentials system with default admin user."""
    if not USERS_FILE.exists():
        default_users = {
            "admin": {
                "password_hash": hash_password("admin123"),
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "active": True
            }
        }
        
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f, indent=2)
        
        # Initialize password history
        history = {
            "admin": [{
                "password_hash": hash_password("admin123"),
                "changed_at": datetime.now().isoformat(),
                "changed_by": "system"
            }]
        }
        
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        print("🔐 Credentials initialized - Default login: admin/admin123")

def load_users():
    """Load users from file."""
    if not USERS_FILE.exists():
        initialize_credentials()
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    """Save users to file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_password_history():
    """Load password history from file."""
    if not HISTORY_FILE.exists():
        return {}
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_password_history(history):
    """Save password history to file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def verify_user(username, password):
    """Verify user credentials."""
    users = load_users()
    if username not in users:
        return False
    
    user = users[username]
    if not user.get('active', True):
        return False
    
    password_hash = hash_password(password)
    return password_hash == user['password_hash']

def update_last_login(username):
    """Update user's last login time."""
    users = load_users()
    if username in users:
        users[username]['last_login'] = datetime.now().isoformat()
        save_users(users)

def change_password(username, old_password, new_password, changed_by=None):
    """Change user password with history tracking."""
    users = load_users()
    
    if username not in users:
        return False, "User not found"
    
    # Verify old password (except for admin changing other users)
    if changed_by != username and changed_by != "admin":
        if not verify_user(username, old_password):
            return False, "Current password is incorrect"
    
    # Check if new password was used before
    history = load_password_history()
    user_history = history.get(username, [])
    new_password_hash = hash_password(new_password)
    
    for entry in user_history[-5:]:  # Check last 5 passwords
        if entry['password_hash'] == new_password_hash:
            return False, "Cannot reuse recent passwords"
    
    # Update password
    users[username]['password_hash'] = new_password_hash
    save_users(users)
    
    # Add to history
    if username not in history:
        history[username] = []
    
    history[username].append({
        "password_hash": new_password_hash,
        "changed_at": datetime.now().isoformat(),
        "changed_by": changed_by or username
    })
    
    # Keep only last 10 password changes
    history[username] = history[username][-10:]
    save_password_history(history)
    
    return True, "Password changed successfully"

def get_user_info(username):
    """Get user information."""
    users = load_users()
    if username in users:
        user_info = users[username].copy()
        user_info.pop('password_hash', None)  # Remove sensitive data
        return user_info
    return None

def get_password_history(username):
    """Get password change history for user."""
    history = load_password_history()
    user_history = history.get(username, [])
    
    # Remove password hashes from history for security
    safe_history = []
    for entry in user_history:
        safe_entry = entry.copy()
        safe_entry.pop('password_hash', None)
        safe_history.append(safe_entry)
    
    return safe_history

def create_user(username, password, role="user", created_by="admin"):
    """Create a new user."""
    users = load_users()
    
    if username in users:
        return False, "User already exists"
    
    users[username] = {
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "created_by": created_by,
        "last_login": None,
        "active": True
    }
    
    save_users(users)
    
    # Initialize password history
    history = load_password_history()
    history[username] = [{
        "password_hash": hash_password(password),
        "changed_at": datetime.now().isoformat(),
        "changed_by": created_by
    }]
    
    save_password_history(history)
    return True, "User created successfully"

def list_users():
    """List all users (without passwords)."""
    users = load_users()
    safe_users = {}
    
    for username, user_data in users.items():
        safe_users[username] = user_data.copy()
        safe_users[username].pop('password_hash', None)
    
    return safe_users

# Initialize on import
initialize_credentials()
