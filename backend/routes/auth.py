from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from config.settings import supabase
from utils.auth import verify_password, hash_password
from utils.subscription import _get_free_plan_id
import random
import string
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

auth_bp = Blueprint('auth', __name__)

# ─────────────────────────────────────────────
#  HELPER — Send OTP email via Gmail SMTP
# ─────────────────────────────────────────────
def send_otp_email(to_email: str, otp: str, name: str = "User"):
    """
    Sends OTP email using Gmail SMTP.
    Set these in your .env / environment:
      EMAIL_ADDRESS = your Gmail address
      EMAIL_PASSWORD = your Gmail App Password (not your real password)
    """
    from_email = os.getenv("EMAIL_ADDRESS")
    app_password = os.getenv("EMAIL_PASSWORD")

    if not from_email or not app_password:
        print("[EMAIL] EMAIL_ADDRESS or EMAIL_PASSWORD not set in environment!")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your Verification Code – Resume Analyser"
        msg["From"]    = from_email
        msg["To"]      = to_email

        html_body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 480px; margin: auto; padding: 32px; border: 1px solid #e5e7eb; border-radius: 8px;">
            <h2 style="color: #1f2937;">Hi {name} 👋</h2>
            <p style="color: #4b5563;">Use the OTP below to verify your email address.</p>
            <div style="font-size: 36px; font-weight: bold; letter-spacing: 10px; color: #4f46e5; margin: 24px 0;">
                {otp}
            </div>
            <p style="color: #9ca3af; font-size: 13px;">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
        </div>
        """

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, app_password)
            server.sendmail(from_email, to_email, msg.as_string())

        print(f"[EMAIL] OTP sent to {to_email}")
        return True

    except Exception as e:
        print(f"[EMAIL] Failed to send OTP: {e}")
        return False


# ─────────────────────────────────────────────
#  HELPER — Generate 6-digit OTP
# ─────────────────────────────────────────────
def generate_otp():
    return ''.join(random.choices(string.digits, k=6))


# ─────────────────────────────────────────────
#  INDEX
# ─────────────────────────────────────────────
@auth_bp.route('/')
def index():
    print(f"[INDEX] session = {dict(session)}")
    if 'user_username' not in session:
        return redirect(url_for('auth.login'))
    if session.get('user_role') == 'admin':
        return redirect(url_for('auth.admin_page'))
    return render_template('index.html')


# ─────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '').strip()

        if not username or not password:
            return render_template('login.html', error='Username and password are required')

        try:
            # 1. Find user in YOUR users table
            result = supabase.table("users").select("*").eq("username", username).execute()

            if not result.data:
                return render_template('login.html', error='Invalid username or password')

            user = result.data[0]

            # 2. Check if email is verified (your own flag now)
            if not user.get('is_verified', False):
                return render_template('login.html', error='Please verify your email before logging in. Check your inbox for the OTP.')

            # 3. Verify password using bcrypt
            if not verify_password(password, user.get('password', '')):
                return render_template('login.html', error='Invalid username or password')

            # 4. Set session
            session.permanent = True
            session['user_username'] = user['username']
            session['user_name']     = user.get('name', username)
            session['user_role']     = user.get('role', 'candidate')
            session.modified = True

            if user.get('role') == 'admin':
                return redirect(url_for('auth.admin_page'))
            return redirect(url_for('auth.index'))

        except Exception as e:
            return render_template('login.html', error=f'Login failed: {str(e)}')

    return render_template('login.html', error=None)


# ─────────────────────────────────────────────
#  ADMIN PAGE
# ─────────────────────────────────────────────
@auth_bp.route('/admin')
def admin_page():
    if 'user_username' not in session:
        return redirect(url_for('auth.login'))
    if session.get('user_role') != 'admin':
        return redirect(url_for('auth.index'))
    return render_template('admin.html')


# ─────────────────────────────────────────────
#  SIGNUP
# ─────────────────────────────────────────────
@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data     = request.get_json()
        name     = (data.get('name')     or '').strip()
        username = (data.get('username') or '').strip()
        email    = (data.get('email')    or '').strip()
        password = (data.get('password') or '').strip()
        role     = (data.get('role')     or 'candidate').strip().lower()

        if role not in ('admin', 'candidate'):
            role = 'candidate'

        if not all([name, username, email, password]):
            return jsonify({'error': 'All fields are required'}), 400

        try:
            # Check duplicate username
            existing = supabase.table("users").select("id").eq("username", username).execute()
            if existing.data:
                return jsonify({'error': 'Username already taken'}), 400

            # Check duplicate email
            existing_email = supabase.table("users").select("id").eq("email", email).execute()
            if existing_email.data:
                return jsonify({'error': 'An account with this email already exists'}), 400

            # Generate OTP and expiry
            otp        = generate_otp()
            otp_expiry = datetime.utcnow() + timedelta(minutes=10)

            # Hash password
            hashed_pw = hash_password(password)

            # Insert user into YOUR users table (is_verified = False until OTP confirmed)
            supabase.table("users").insert({
                "name":           name,
                "username":       username,
                "email":          email,
                "password":       hashed_pw,
                "role":           role,
                "is_verified":    False,
                "otp_code":       otp,
                "otp_expires_at": otp_expiry.isoformat()
            }).execute()

            # Send OTP email (your own Gmail SMTP — no Supabase!)
            sent = send_otp_email(email, otp, name)
            if not sent:
                return jsonify({'error': 'Account created but failed to send OTP. Contact support.'}), 500

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('signup.html')


# ─────────────────────────────────────────────
#  VERIFY PAGE
# ─────────────────────────────────────────────
@auth_bp.route('/verify')
def verify_page():
    return render_template('verify.html')


# ─────────────────────────────────────────────
#  VERIFY OTP  ← completely your own now
# ─────────────────────────────────────────────
@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data  = request.get_json()
    email = (data.get('email') or '').strip()
    otp   = (data.get('otp')   or '').strip()

    if not email or not otp:
        return jsonify({'error': 'Email and OTP are required'}), 400

    try:
        # Get user by email
        result = supabase.table("users").select("*").eq("email", email).execute()

        if not result.data:
            return jsonify({'error': 'No account found with this email'}), 404

        user = result.data[0]

        # Check if already verified
        if user.get('is_verified'):
            return jsonify({'success': True, 'message': 'Already verified'})

        # Check OTP match
        if user.get('otp_code') != otp:
            return jsonify({'error': 'Invalid OTP. Please try again.'}), 400

        # Check OTP expiry
        otp_expiry = user.get('otp_expires_at')
        if otp_expiry:
            expiry_dt = datetime.fromisoformat(otp_expiry.replace('Z', ''))
            if datetime.utcnow() > expiry_dt:
                return jsonify({'error': 'OTP has expired. Please request a new one.'}), 400

        # ✅ Mark user as verified and clear OTP
        supabase.table("users").update({
            "is_verified":    True,
            "otp_code":       None,
            "otp_expires_at": None
        }).eq("email", email).execute()

        # Auto-assign Free plan
        try:
            free_id = _get_free_plan_id()
            username = user.get('username')
            if free_id and username:
                supabase.table("user_subscriptions").insert({
                    "username":     username,
                    "plan_id":      free_id,
                    "status":       "active",
                    "resumes_used": 0,
                    "mcq_used":     0,
                }).execute()
        except Exception as sub_err:
            print(f"[verify] Could not create subscription: {sub_err}")

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  RESEND OTP  ← your own now
# ─────────────────────────────────────────────
@auth_bp.route('/resend-otp', methods=['POST'])
def resend_otp():
    data  = request.get_json()
    email = (data.get('email') or '').strip()

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    try:
        result = supabase.table("users").select("*").eq("email", email).execute()

        if not result.data:
            return jsonify({'error': 'No account found with this email'}), 404

        user = result.data[0]

        if user.get('is_verified'):
            return jsonify({'error': 'Email is already verified'}), 400

        # Generate new OTP
        otp        = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=10)

        # Update OTP in DB
        supabase.table("users").update({
            "otp_code":       otp,
            "otp_expires_at": otp_expiry.isoformat()
        }).eq("email", email).execute()

        # Send new OTP email
        sent = send_otp_email(email, otp, user.get('name', 'User'))
        if not sent:
            return jsonify({'error': 'Failed to resend OTP. Try again later.'}), 500

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
#  FORGOT PASSWORD  ← your own now
# ─────────────────────────────────────────────
@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        data  = request.get_json()
        email = (data.get('email') or '').strip()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        try:
            result = supabase.table("users").select("*").eq("email", email).execute()

            if not result.data:
                # Don't reveal if email exists or not (security best practice)
                return jsonify({'success': True})

            user = result.data[0]

            # Generate OTP for password reset
            otp        = generate_otp()
            otp_expiry = datetime.utcnow() + timedelta(minutes=10)

            supabase.table("users").update({
                "otp_code":       otp,
                "otp_expires_at": otp_expiry.isoformat()
            }).eq("email", email).execute()

            send_otp_email(email, otp, user.get('name', 'User'))

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('forgot_password.html')


# ─────────────────────────────────────────────
#  RESET PASSWORD  ← your own now
# ─────────────────────────────────────────────
@auth_bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        data         = request.get_json()
        email        = (data.get('email')    or '').strip()
        otp          = (data.get('otp')      or '').strip()
        new_password = (data.get('password') or '').strip()

        if not email or not otp or not new_password:
            return jsonify({'error': 'Email, OTP, and new password are required'}), 400

        if len(new_password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        try:
            result = supabase.table("users").select("*").eq("email", email).execute()

            if not result.data:
                return jsonify({'error': 'No account found'}), 404

            user = result.data[0]

            # Verify OTP
            if user.get('otp_code') != otp:
                return jsonify({'error': 'Invalid OTP'}), 400

            # Check expiry
            otp_expiry = user.get('otp_expires_at')
            if otp_expiry:
                expiry_dt = datetime.fromisoformat(otp_expiry.replace('Z', ''))
                if datetime.utcnow() > expiry_dt:
                    return jsonify({'error': 'OTP has expired. Please request a new one.'}), 400

            # Update password and clear OTP
            hashed_pw = hash_password(new_password)
            supabase.table("users").update({
                "password":       hashed_pw,
                "otp_code":       None,
                "otp_expires_at": None
            }).eq("email", email).execute()

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('reset_password.html')


# ─────────────────────────────────────────────
#  LOGOUT
# ─────────────────────────────────────────────
@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))


# ─────────────────────────────────────────────
#  OAUTH CALLBACK (Google login — unchanged)
# ─────────────────────────────────────────────
@auth_bp.route('/oauth-callback', methods=['POST'])
def oauth_callback():
    import requests as http_requests
    from config.settings import SUPABASE_URL, SUPABASE_KEY

    data         = request.get_json()
    access_token = data.get('access_token')

    if not access_token:
        return jsonify({"error": "No token provided"}), 400

    resp = http_requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {access_token}"
        }
    )

    if resp.status_code != 200:
        return jsonify({"error": "Invalid token"}), 401

    user      = resp.json()
    email     = user.get('email', '')
    user_meta = user.get('user_metadata', {})
    name      = user_meta.get('full_name') or user_meta.get('name') or email.split('@')[0]

    existing = supabase.table("users").select("*").eq("email", email).execute()

    if existing.data:
        db_user  = existing.data[0]
        username = db_user['username']
        role     = db_user.get('role', 'candidate')
    else:
        username = email.split('@')[0].replace('.', '_').lower()
        check    = supabase.table("users").select("id").eq("username", username).execute()
        if check.data:
            username = username + '_' + str(user['id'])[:4]

        supabase.table("users").insert({
            "name":        name,
            "username":    username,
            "email":       email,
            "password":    "",
            "role":        "candidate",
            "is_verified": True   # OAuth users are already verified via Google
        }).execute()
        role = "candidate"

        try:
            free_id = _get_free_plan_id()
            if free_id:
                supabase.table("user_subscriptions").insert({
                    "username":     username,
                    "plan_id":      free_id,
                    "status":       "active",
                    "resumes_used": 0,
                    "mcq_used":     0,
                }).execute()
        except Exception as sub_err:
            print(f"[oauth] Could not create subscription: {sub_err}")

    session.permanent = True
    session['user_username'] = username
    session['user_name']     = name
    session['user_role']     = role
    session.modified = True

    redirect_url = '/admin' if role == 'admin' else '/'
    return jsonify({"redirect": redirect_url})


# ─────────────────────────────────────────────
#  API /me
# ─────────────────────────────────────────────
@auth_bp.route('/api/me', methods=['GET'])
def get_me():
    if 'user_username' not in session:
        return jsonify({'authenticated': False}), 401
    return jsonify({
        'authenticated': True,
        'username':  session.get('user_username'),
        'name':      session.get('user_name'),
        'role':      session.get('user_role'),
        'job_role':  session.get('predicted_job_role', ''),
    })