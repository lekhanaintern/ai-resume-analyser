from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from config.settings import supabase, SUPABASE_URL, SUPABASE_KEY, SUPABASE_SERVICE_KEY
from utils.auth import verify_password, hash_password, _upgrade_password_hash
from utils.subscription import _get_free_plan_id

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/')
def index():
    print(f"[INDEX] session = {dict(session)}")
    if 'user_username' not in session:
        return redirect(url_for('auth.login'))
    if session.get('user_role') == 'admin':
        return redirect(url_for('auth.admin_page'))
    return render_template('index.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        import requests as http_requests
        username = (request.form.get('username') or '').strip()
        password = (request.form.get('password') or '').strip()
        if not username or not password:
            return render_template('login.html', error='Username and password required')
        try:
            result = supabase.table("users").select("*").eq("username", username).execute()
            if not result.data:
                return render_template('login.html', error='Invalid username or password')
            user = result.data[0]
            stored_pw = user.get('password', '')
            is_legacy = not (stored_pw.startswith('$2b$') or stored_pw.startswith('$2a$') or stored_pw.startswith('$2y$'))
            if not verify_password(password, stored_pw):
                return render_template('login.html', error='Invalid username or password')

            # Check if email is verified in Supabase Auth
            email = user.get('email', '')
            auth_resp = http_requests.get(
                f"{SUPABASE_URL}/auth/v1/admin/users",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY != "PASTE_YOUR_SERVICE_ROLE_KEY_HERE" else SUPABASE_KEY,
                    "Authorization": f"Bearer {(SUPABASE_SERVICE_KEY if SUPABASE_SERVICE_KEY != 'PASTE_YOUR_SERVICE_ROLE_KEY_HERE' else SUPABASE_KEY)}",
                    "Content-Type": "application/json"
                },
                params={"filter": f"email.eq.{email}"}
            )
            if auth_resp.status_code == 200:
                auth_users = auth_resp.json().get('users', [])
                if auth_users:
                    confirmed = auth_users[0].get('email_confirmed_at')
                    if not confirmed:
                        return render_template('login.html', error='Please verify your email before logging in. Check your inbox for the OTP.')

            # Auto-upgrade legacy plain-text password to bcrypt on successful login
            if is_legacy:
                _upgrade_password_hash(username, password)
            session.permanent = True
            session['user_username'] = username
            session['user_name']     = user.get('name', username)
            session['user_role']     = user.get('role', 'candidate')
            session.modified = True
            if user.get('role') == 'admin':
                return redirect(url_for('auth.admin_page'))
            return redirect(url_for('auth.index'))
        except Exception as e:
            return render_template('login.html', error=f'Login failed: {str(e)}')
    return render_template('login.html', error=None)


@auth_bp.route('/admin')
def admin_page():
    if 'user_username' not in session:
        return redirect(url_for('auth.login'))
    if session.get('user_role') != 'admin':
        return redirect(url_for('auth.index'))
    return render_template('admin.html')


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        import requests as http_requests
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
            existing = supabase.table("users").select("id").eq("username", username).execute()
            if existing.data:
                return jsonify({'error': 'Username already taken'}), 400
            existing_email = supabase.table("users").select("id").eq("email", email).execute()
            if existing_email.data:
                return jsonify({'error': 'An account with this email already exists'}), 400

            # Register via Supabase Auth — this triggers the OTP email
            resp = http_requests.post(
                f"{SUPABASE_URL}/auth/v1/signup",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Content-Type": "application/json"
                },
                json={"email": email, "password": password, "options": {"emailRedirectTo": None, "data": {"name": name}}},
                timeout=15
            )
            print(f"[signup] status={resp.status_code}")
            print(f"[signup] body={resp.text[:400]}")

            if resp.status_code not in (200, 201):
                body = resp.json()
                err = body.get('msg') or body.get('message') or body.get('error_description') or body.get('error') or 'Signup failed. Please try again.'
                return jsonify({'error': err}), 400

            # Insert extra user info into your custom users table
            hashed_pw = hash_password(password)
            supabase.table("users").insert({
                "name": name, "username": username,
                "email": email, "password": hashed_pw, "role": role
            }).execute()

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('signup.html')
@auth_bp.route('/verify')
def verify_page():
    return render_template('verify.html')

@auth_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    import requests as http_requests
    data  = request.get_json()
    email = (data.get('email') or '').strip()
    otp   = (data.get('otp') or '').strip()

    if not email or not otp:
        return jsonify({'error': 'Email and OTP are required'}), 400

    try:
        # Verify OTP with Supabase
        resp = http_requests.post(
            f"{SUPABASE_URL}/auth/v1/verify",
            headers={
                "apikey":        SUPABASE_KEY,
                "Content-Type":  "application/json"
            },
            json={"type": "signup", "email": email, "token": otp}
        )

        if resp.status_code == 200:
            return jsonify({'success': True})
        else:
            err = resp.json().get('error_description') or resp.json().get('msg') or 'Invalid OTP'
            return jsonify({'error': err}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/resend-otp', methods=['POST'])
def resend_otp():
    import requests as http_requests
    data  = request.get_json()
    email = (data.get('email') or '').strip()

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    try:
        resp = http_requests.post(
            f"{SUPABASE_URL}/auth/v1/resend",
            headers={
                "apikey":       SUPABASE_KEY,
                "Content-Type": "application/json"
            },
            json={"type": "signup", "email": email}
        )

        if resp.status_code in (200, 204):
            return jsonify({'success': True})
        else:
            err = resp.json().get('error_description') or 'Could not resend OTP'
            return jsonify({'error': err}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        import requests as http_requests
        data  = request.get_json()
        email = (data.get('email') or '').strip()
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        try:
            resp = http_requests.post(
                f"{SUPABASE_URL}/auth/v1/recover",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Content-Type": "application/json"
                },
                json={"email": email}
            )
            if resp.status_code in (200, 204):
                return jsonify({'success': True})
            else:
                err = resp.json().get('msg') or 'Could not send reset email'
                return jsonify({'error': err}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('forgot_password.html')


@auth_bp.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        data        = request.get_json()
        email       = (data.get('email')    or '').strip()
        new_password = (data.get('password') or '').strip()
        if not email or not new_password:
            return jsonify({'error': 'Email and new password are required'}), 400
        if len(new_password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        try:
            hashed_pw = hash_password(new_password)
            supabase.table("users").update({"password": hashed_pw}).eq("email", email).execute()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return render_template('reset_password.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))

@auth_bp.route('/oauth-callback', methods=['POST'])
def oauth_callback():
    import requests as http_requests
    data         = request.get_json()
    access_token = data.get('access_token')

    if not access_token:
        return jsonify({"error": "No token provided"}), 400

    # Verify token with Supabase and get user info
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

    # Check if user already exists in your users table
    existing = supabase.table("users").select("*").eq("email", email).execute()

    if existing.data:
        db_user = existing.data[0]
        username = db_user['username']
        role     = db_user.get('role', 'candidate')
    else:
        # Auto-create a new user from OAuth
        username = email.split('@')[0].replace('.', '_').lower()
        # Make sure username is unique
        check = supabase.table("users").select("id").eq("username", username).execute()
        if check.data:
            username = username + '_' + str(user['id'])[:4]

        supabase.table("users").insert({
            "name":     name,
            "username": username,
            "email":    email,
            "password": "",  # No password for OAuth users
            "role":     "candidate"
        }).execute()
        role = "candidate"

        # Give them a free plan
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

    # Set session
    session.permanent = True
    session['user_username'] = username
    session['user_name']     = name
    session['user_role']     = role
    session.modified = True

    redirect_url = '/admin' if role == 'admin' else '/'
    return jsonify({"redirect": redirect_url})


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