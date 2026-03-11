import bcrypt
from functools import wraps
from flask import session, request, jsonify, redirect, url_for


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain: str, stored: str) -> bool:
    if not plain or not stored:
        return False
    stored_bytes = stored.encode('utf-8')
    if stored.startswith('$2b$') or stored.startswith('$2a$') or stored.startswith('$2y$'):
        try:
            return bcrypt.checkpw(plain.encode('utf-8'), stored_bytes)
        except Exception:
            return False
    return plain == stored


def _upgrade_password_hash(username: str, plain: str):
    try:
        from config.settings import supabase
        new_hash = hash_password(plain)
        supabase.table("users").update({"password": new_hash}).eq("username", username).execute()
        print(f"[auth] Upgraded password hash for '{username}' to bcrypt.")
    except Exception as e:
        print(f"[auth] Could not upgrade password hash for '{username}': {e}")


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('user_username') or session.get('user_role') != 'admin':
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Admin access required.', 'auth_error': True}), 403
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function