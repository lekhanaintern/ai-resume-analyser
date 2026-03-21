import os
from flask import Blueprint, request, jsonify
from config.settings import supabase, _cache_get, _cache_set, _cache_clear
from utils.auth import admin_required

admin_bp = Blueprint('admin', __name__)


# ============================================================
# ADMIN API ROUTES
# ============================================================
@admin_bp.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    try:
        result = supabase.table("users").select("id, name, username, email, role", count='exact').execute()
        total  = result.count if result.count is not None else len(result.data or [])
        return jsonify({'success': True, 'users': result.data or [], 'total': total})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/delete-user', methods=['POST'])
@admin_required
def admin_delete_user():
    try:
        user_id = request.get_json().get('id')
        supabase.table("users").delete().eq("id", user_id).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _fetch_role_counts():
    seen_roles = set()
    offset = 0
    while True:
        res   = supabase.table("mcq_questions").select("job_role").range(offset, offset + 999).execute()
        batch = res.data or []
        for row in batch:
            r = row.get('job_role')
            if r:
                seen_roles.add(r)
        if len(batch) < 1000:
            break
        offset += 1000
    all_roles = list(seen_roles)

    role_counts = {}
    for role in all_roles:
        r = supabase.table("mcq_questions").select("id", count='exact').eq("job_role", role).execute()
        role_counts[role] = r.count if r.count is not None else 0

    diff_counts = {}
    for diff in ['easy', 'medium', 'hard']:
        r = supabase.table("mcq_questions").select("id", count='exact').eq("difficulty", diff).execute()
        diff_counts[diff] = r.count if r.count is not None else 0

    roles_sorted = sorted(
        [{'role': k, 'count': v} for k, v in role_counts.items()],
        key=lambda x: x['count'], reverse=True
    )
    return roles_sorted, diff_counts


@admin_bp.route('/api/admin/dashboard', methods=['GET'])
@admin_required
def admin_dashboard():
    try:
        try:
            supabase.table("mcq_results").update({"username": "unknown"}).is_("username", "null").execute()
            supabase.table("mcq_results").update({"username": "unknown"}).eq("username", "").execute()
        except Exception:
            pass

        cached = _cache_get('dashboard', ttl=60)
        if cached:
            return jsonify(cached)

        users_r    = supabase.table("users").select("id, name, username, email, role", count='exact').execute()
        user_list  = users_r.data or []
        user_total = users_r.count if users_r.count is not None else len(user_list)

        total_r    = supabase.table("mcq_questions").select("*", count='exact').execute()
        active_r   = supabase.table("mcq_questions").select("*", count='exact').eq("status", "active").execute()
        inactive_r = supabase.table("mcq_questions").select("*", count='exact').eq("status", "inactive").execute()
        q_total    = total_r.count    if total_r.count    is not None else len(total_r.data or [])
        q_active   = active_r.count   if active_r.count   is not None else len(active_r.data or [])
        q_inactive = inactive_r.count if inactive_r.count is not None else len(inactive_r.data or [])

        roles_sorted, diff_counts = _fetch_role_counts()

        results_r = supabase.table("mcq_results").select("*").order("created_at", desc=True).limit(5).execute()
        recent    = results_r.data or []

        payload = {
            'success':    True,
            'user_total': user_total,
            'users':      user_list,
            'q_total':    q_total,
            'q_active':   q_active,
            'q_inactive': q_inactive,
            'roles':      roles_sorted,
            'difficulty': diff_counts,
            'results':    recent,
        }
        _cache_set('dashboard', payload)
        return jsonify(payload)
    except Exception as e:
        print(f"[admin_dashboard] ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@admin_bp.route('/api/admin/question-stats', methods=['GET'])
@admin_required
def admin_question_stats():
    try:
        cached = _cache_get('question_stats', ttl=300)
        if cached:
            return jsonify(cached)
        total_r    = supabase.table("mcq_questions").select("*", count='exact').execute()
        active_r   = supabase.table("mcq_questions").select("*", count='exact').eq("status", "active").execute()
        inactive_r = supabase.table("mcq_questions").select("*", count='exact').eq("status", "inactive").execute()
        total    = total_r.count    if total_r.count    is not None else len(total_r.data or [])
        active   = active_r.count   if active_r.count   is not None else len(active_r.data or [])
        inactive = inactive_r.count if inactive_r.count is not None else len(inactive_r.data or [])
        payload = {'success': True, 'total': total, 'active': active, 'inactive': inactive}
        _cache_set('question_stats', payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/role-stats', methods=['GET'])
@admin_required
def admin_role_stats():
    try:
        cached = _cache_get('role_stats', ttl=60)
        if cached:
            return jsonify(cached)
        roles_sorted, diff_counts = _fetch_role_counts()
        payload = {'success': True, 'roles': roles_sorted, 'difficulty': diff_counts}
        _cache_set('role_stats', payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/questions', methods=['GET'])
@admin_required
def admin_get_questions():
    try:
        all_rows  = []
        page_size = 1000
        offset    = 0
        while True:
            result = supabase.table("mcq_questions").select(
                "id, job_role, question, options, correct_answer, difficulty, explanation, status"
            ).range(offset, offset + page_size - 1).limit(page_size).execute()
            batch = result.data or []
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return jsonify({'success': True, 'questions': all_rows, 'total': len(all_rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/toggle-question-status', methods=['POST'])
@admin_required
def admin_toggle_status():
    try:
        data   = request.get_json()
        qid    = data.get('id')
        status = data.get('status')
        if status not in ('active', 'inactive'):
            return jsonify({'error': 'Invalid status'}), 400
        supabase.table("mcq_questions").update({"status": status}).eq("id", qid).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/delete-question', methods=['POST'])
@admin_required
def admin_delete_question():
    try:
        qid = request.get_json().get('id')
        supabase.table("mcq_questions").delete().eq("id", qid).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/add-question', methods=['POST'])
@admin_required
def admin_add_question():
    try:
        data = request.get_json()
        supabase.table("mcq_questions").insert({
            "job_role":       data.get('job_role'),
            "question":       data.get('question'),
            "options":        data.get('options'),
            "correct_answer": data.get('correct_answer'),
            "difficulty":     data.get('difficulty', 'medium'),
            "explanation":    data.get('explanation', ''),
            "status":         "active"
        }).execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# BULK UPLOAD QUESTIONS
# POST /api/admin/bulk-upload-questions
# Body: { questions: [ { job_role, question, options, correct_answer, difficulty, explanation } ] }
# Returns: { success, inserted, failed, errors }
# ============================================================
@admin_bp.route('/api/admin/bulk-upload-questions', methods=['POST'])
@admin_required
def admin_bulk_upload_questions():
    VALID_ROLES = {
        'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
        'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
        'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
        'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
        'BANKING', 'ARTS', 'AVIATION', 'DATA-SCIENCE', 'WEB-DEVELOPER', 'DEFAULT'
    }
    VALID_DIFFICULTIES = {'easy', 'medium', 'hard'}

    try:
        data      = request.get_json(force=True, silent=True)
        questions = (data or {}).get('questions', [])

        if not questions or not isinstance(questions, list):
            return jsonify({'error': 'questions array is required'}), 400
        if len(questions) > 500:
            return jsonify({'error': 'Maximum 500 questions per batch'}), 400

        to_insert  = []
        errors     = []

        for idx, q in enumerate(questions):
            row_num  = idx + 1
            job_role = (q.get('job_role') or '').strip().upper()
            question = (q.get('question') or '').strip()
            options  = q.get('options')
            correct  = (q.get('correct_answer') or '').strip()
            diff     = (q.get('difficulty') or 'medium').strip().lower()
            expl     = (q.get('explanation') or '').strip()

            # Validate
            if not question or len(question) < 5:
                errors.append(f'Row {row_num}: question too short or missing')
                continue
            if job_role not in VALID_ROLES:
                errors.append(f'Row {row_num}: invalid job_role "{job_role}"')
                continue
            if not isinstance(options, list) or len(options) != 4:
                errors.append(f'Row {row_num}: options must be a list of 4')
                continue
            if any(not str(o).strip() for o in options):
                errors.append(f'Row {row_num}: all 4 options must be non-empty')
                continue
            if not correct:
                errors.append(f'Row {row_num}: correct_answer is required')
                continue
            if correct not in options:
                errors.append(f'Row {row_num}: correct_answer not found in options')
                continue
            if diff not in VALID_DIFFICULTIES:
                errors.append(f'Row {row_num}: difficulty must be easy/medium/hard')
                continue

            to_insert.append({
                "job_role":       job_role,
                "question":       question,
                "options":        [str(o).strip() for o in options],
                "correct_answer": correct,
                "difficulty":     diff,
                "explanation":    expl,
                "status":         "active"
            })

        if not to_insert:
            return jsonify({'success': False, 'inserted': 0, 'failed': len(questions), 'errors': errors}), 400

        # Insert in chunks of 100 to avoid Supabase limits
        CHUNK = 100
        inserted = 0
        insert_errors = []

        for i in range(0, len(to_insert), CHUNK):
            chunk = to_insert[i:i + CHUNK]
            try:
                supabase.table("mcq_questions").insert(chunk).execute()
                inserted += len(chunk)
            except Exception as chunk_err:
                insert_errors.append(str(chunk_err))
                # Try one-by-one as fallback for this chunk
                for single in chunk:
                    try:
                        supabase.table("mcq_questions").insert(single).execute()
                        inserted += 1
                    except Exception as single_err:
                        errors.append(f'Insert error: {str(single_err)[:80]}')

        _cache_clear()  # Invalidate admin stats cache

        return jsonify({
            'success':  True,
            'inserted': inserted,
            'failed':   len(questions) - inserted,
            'errors':   (errors + insert_errors)[:20]  # cap at 20 error messages
        })

    except Exception as e:
        print(f'[bulk-upload] ERROR: {e}')
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/results', methods=['GET'])
@admin_required
def admin_get_results():
    try:
        limit = int(request.args.get('limit', 50))
        cache_key = f'results_{limit}'
        cached = _cache_get(cache_key, ttl=30)
        if cached:
            return jsonify(cached)
        result = supabase.table("mcq_results").select("*").order("created_at", desc=True).limit(limit).execute()
        payload = {'success': True, 'results': result.data or []}
        _cache_set(cache_key, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# EMAIL CONFIGURATION ROUTES
# GET  /api/admin/email-config  — return current EMAIL_ADDRESS (masked)
# POST /api/admin/email-config  — write EMAIL_ADDRESS + EMAIL_PASSWORD to .env
# POST /api/admin/email-config/test — send a test email to the configured address
# ============================================================

def _find_env_path():
    """Walk up from this file to find the .env file."""
    base = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # look up to 4 levels up
        candidate = os.path.join(base, '.env')
        if os.path.exists(candidate):
            return candidate
        base = os.path.dirname(base)
    # If not found, create in project root (2 levels up from routes/)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, '.env')


def _read_env_file(path):
    """Read .env file into a dict."""
    pairs = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    pairs[key.strip()] = val.strip().strip('"').strip("'")
    return pairs


def _write_env_file(path, pairs):
    """Write dict back to .env, preserving comments and order."""
    existing_lines = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()

    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            if key in pairs:
                new_lines.append(f'{key}={pairs[key]}\n')
                updated_keys.add(key)
                continue
        new_lines.append(line if line.endswith('\n') else line + '\n')

    # Append any new keys that weren't in the file
    for key, val in pairs.items():
        if key not in updated_keys:
            new_lines.append(f'{key}={val}\n')

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


@admin_bp.route('/api/admin/email-config', methods=['GET'])
@admin_required
def get_email_config():
    try:
        email = os.getenv('EMAIL_ADDRESS', '')
        # Mask the email for display: show first 3 chars + *** + @domain
        masked = ''
        if email:
            parts = email.split('@')
            if len(parts) == 2:
                local = parts[0]
                masked = local[:3] + '***@' + parts[1] if len(local) > 3 else email
            else:
                masked = email
        return jsonify({'success': True, 'email': email, 'masked': masked})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/email-config', methods=['POST'])
@admin_required
def save_email_config():
    try:
        data     = request.get_json()
        email    = (data.get('email')    or '').strip()
        password = (data.get('password') or '').strip().replace(' ', '')

        if not email or not password:
            return jsonify({'error': 'Both email and password are required'}), 400
        if '@gmail.com' not in email:
            return jsonify({'error': 'Only Gmail addresses are supported'}), 400
        if len(password) < 16:
            return jsonify({'error': 'App Password must be at least 16 characters'}), 400

        env_path = _find_env_path()
        _write_env_file(env_path, {
            'EMAIL_ADDRESS':  email,
            'EMAIL_PASSWORD': password,
        })

        # Hot-reload into current process so it takes effect immediately
        os.environ['EMAIL_ADDRESS']  = email
        os.environ['EMAIL_PASSWORD'] = password

        print(f'[email-config] Updated EMAIL_ADDRESS to {email} in {env_path}')
        return jsonify({'success': True})

    except Exception as e:
        print(f'[email-config] ERROR: {e}')
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/admin/email-config/test', methods=['POST'])
@admin_required
def test_email_config():
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        from_email   = os.getenv('EMAIL_ADDRESS', '')
        app_password = os.getenv('EMAIL_PASSWORD', '')

        if not from_email or not app_password:
            return jsonify({'error': 'No email configuration found. Please save your settings first.'}), 400

        msg = MIMEMultipart('alternative')
        msg['Subject'] = '✅ Test Email — AI Resume Analyser'
        msg['From']    = from_email
        msg['To']      = from_email  # send test to yourself

        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;padding:32px;border:1px solid #e5e7eb;border-radius:8px;">
            <h2 style="color:#1f2937;">✅ Email Configuration Working!</h2>
            <p style="color:#4b5563;">Your Gmail SMTP is correctly set up for AI Resume Analyser.</p>
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:16px;margin:16px 0;">
                <strong style="color:#065f46;">📧 Sending from:</strong> {from_email}
            </div>
            <p style="color:#9ca3af;font-size:13px;">OTP emails will be sent from this address when users sign up or reset their password.</p>
        </div>
        """
        msg.attach(MIMEText(html, 'html'))

        # Try port 465 first, fall back to 587
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as server:
                server.login(from_email, app_password)
                server.sendmail(from_email, from_email, msg.as_string())
        except Exception:
            with smtplib.SMTP('smtp.gmail.com', 587, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(from_email, app_password)
                server.sendmail(from_email, from_email, msg.as_string())

        return jsonify({'success': True, 'sent_to': from_email})

    except Exception as e:
        print(f'[email-config/test] ERROR: {e}')
        return jsonify({'error': str(e)}), 500