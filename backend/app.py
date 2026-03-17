import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, session, render_template, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
from routes.auth import auth_bp
from routes.subscriptions import subscriptions_bp
from routes.resume import resume_bp
from routes.mcq import mcq_bp
from routes.admin import admin_bp

load_dotenv()

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app, supports_credentials=True)

# ── Secret key loaded from environment ───────────────────────────────────────
_secret_key = os.getenv("FLASK_SECRET_KEY")
if not _secret_key:
    raise EnvironmentError(
        "FLASK_SECRET_KEY is not set. Add it to your .env file. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _secret_key

app.config['SESSION_COOKIE_SAMESITE']    = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY']    = True
app.config['SESSION_COOKIE_SECURE']     = False
app.config['SESSION_COOKIE_NAME']       = 'resume_session'
app.config['SESSION_PERMANENT']         = True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400
app.config['MAX_CONTENT_LENGTH']        = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS']         = ['.pdf', '.docx']


# ── PAGE ROUTES ───────────────────────────────────────────────────────────────

@app.route('/terms')
def terms_page():
    return render_template('terms.html')


@app.route('/login')
def login_page():
    if 'user_username' in session:
        return redirect(url_for('index_page'))
    return render_template('login.html')


@app.route('/signup')
def signup_page():
    if 'user_username' in session:
        return redirect(url_for('index_page'))
    return render_template('signup.html')


@app.route('/')
def index_page():
    if 'user_username' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html',
        user_name=session.get('user_name', 'Candidate'),
        user_username=session.get('user_username', ''),
        user_email=session.get('user_email', ''),
    )


@app.route('/admin')
def admin_page():
    if 'user_username' not in session:
        return redirect(url_for('login_page'))
    return render_template('admin.html',
        user_name=session.get('user_name', 'Candidate'),
        user_username=session.get('user_username', ''),
        user_email=session.get('user_email', ''),
    )


# ── API AUTH GUARD ────────────────────────────────────────────────────────────

@app.before_request
def api_auth_guard():
    if request.path.startswith('/api/'):
        if request.path == '/api/health':
            return None
        if 'user_username' not in session:
            return jsonify({
                'error': 'Session expired. Please refresh the page and log in again.',
                'auth_error': True
            }), 401
    return None


# ── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413


@app.errorhandler(404)
def page_not_found(error):
    return redirect(url_for('login_page'))


# ── BLUEPRINTS ────────────────────────────────────────────────────────────────

app.register_blueprint(auth_bp)
app.register_blueprint(subscriptions_bp)
app.register_blueprint(resume_bp)
app.register_blueprint(mcq_bp)
app.register_blueprint(admin_bp)


if __name__ == '__main__':
    print("=" * 60)
    print("Resume Analysis API — NLP/ML Powered")
    print("=" * 60)
    print("API running at: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')