import random
from datetime import datetime
from flask import Blueprint, request, jsonify, session, render_template
from config.settings import supabase
from utils.auth import admin_required
from utils.subscription import check_limit, increment_usage

mcq_bp = Blueprint('mcq', __name__)


# ============================================================
# MCQ ENDPOINTS
# ============================================================
@mcq_bp.route('/api/get-mcq-test', methods=['GET'])
def get_mcq_test():
    try:
        username = session.get('user_username', '')
        if username:
            allowed, reason = check_limit(username, 'mcq')
            if not allowed:
                return jsonify({'error': reason, 'limit_exceeded': True}), 403

        job_role = request.args.get('role') or session.get('predicted_job_role', 'DEFAULT')

        if request.args.get('role'):
            session['predicted_job_role'] = job_role
            session.modified = True

        seen_ids = session.get('seen_question_ids', [])
        if len(seen_ids) > 500:
            seen_ids = []
            session['seen_question_ids'] = []

        def fetch_questions(role):
            q = supabase.table("mcq_questions") \
                .select("id, question, options, difficulty, correct_answer, explanation") \
                .eq("status", "active") \
                .eq("job_role", role) \
                .execute()
            rows = q.data or []
            rows = [r for r in rows if r['id'] not in seen_ids]
            if len(rows) < 10:
                all_q = supabase.table("mcq_questions") \
                    .select("id, question, options, difficulty, correct_answer, explanation") \
                    .eq("status", "active") \
                    .eq("job_role", role) \
                    .execute()
                rows = all_q.data or []
            random.shuffle(rows)
            return rows[:10]

        questions = fetch_questions(job_role)
        if not questions:
            questions = fetch_questions('DEFAULT')

        if not questions:
            return jsonify({'error': f'No active questions found for role: {job_role}'}), 404

        new_ids = [q['id'] for q in questions]
        session['seen_question_ids'] = seen_ids + new_ids
        session['test_question_ids'] = new_ids
        session['test_start_time']   = datetime.now().isoformat()
        session.modified = True

        if username:
            increment_usage(username, 'mcq')

        safe_questions = [{
            'id':         q['id'],
            'question':   q['question'],
            'options':    q['options'],
            'difficulty': q.get('difficulty', 'medium'),
        } for q in questions]

        return jsonify({
            'success':         True,
            'job_role':        session.get('raw_predicted_role', job_role),
            'db_role':         job_role,
            'questions':       safe_questions,
            'total_questions': len(safe_questions),
            'username':        session.get('user_username', ''),
        })
    except Exception as e:
        print(f"[get-mcq-test] ERROR: {e}")
        return jsonify({'error': str(e)}), 500


@mcq_bp.route('/api/submit-test', methods=['POST'])
def submit_test():
    try:
        data         = request.get_json()
        answers      = data.get('answers', {})
        question_ids = data.get('question_ids') or session.get('test_question_ids', [])
        submitted_username = (data.get('username') or '').strip()
        if not submitted_username:
            submitted_username = session.get('user_username', '').strip()
        if not submitted_username:
            submitted_username = 'anonymous'

        submitted_job_role = (data.get('job_role') or '').strip()
        if not submitted_job_role:
            submitted_job_role = session.get('predicted_job_role', '').strip()
        if not submitted_job_role:
            submitted_job_role = 'UNKNOWN'
        if not question_ids:
            return jsonify({'error': 'No active test found. Please start a new test.'}), 400

        results        = []
        correct_count  = 0
        total_questions = len(question_ids)

        q_res   = supabase.table("mcq_questions") \
            .select("id, question, options, correct_answer, explanation") \
            .in_("id", question_ids) \
            .execute()
        q_map = {q['id']: q for q in (q_res.data or [])}

        for question_id in question_ids:
            question_data  = q_map.get(question_id) or q_map.get(str(question_id)) or {}
            user_answer    = answers.get(str(question_id))
            correct_answer = question_data.get('correct_answer', '')
            is_correct     = (user_answer == correct_answer)
            if is_correct:
                correct_count += 1
            results.append({
                'question_id':    question_id,
                'question':       question_data.get('question', ''),
                'options':        question_data.get('options', []),
                'user_answer':    user_answer,
                'correct_answer': correct_answer,
                'is_correct':     is_correct,
                'explanation':    question_data.get('explanation', '')
            })

        score_pct = (correct_count / total_questions) * 100

        try:
            supabase.table("mcq_results").insert({
                "username":         submitted_username,
                "job_role":         submitted_job_role,
                "score_percentage": round(score_pct, 2),
                "correct_answers":  correct_count,
                "total_questions":  total_questions,
            }).execute()
        except Exception as save_err:
            print(f"[WARN] Could not save test result: {save_err}")

        return jsonify({
            'success': True,
            'score': score_pct,
            'correct_answers': correct_count,
            'wrong_answers': total_questions - correct_count,
            'total_questions': total_questions,
            'results': results,
            'passed': score_pct >= 60
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mcq_bp.route('/api/get-test-history', methods=['GET'])
def get_test_history():
    try:
        response = supabase.table("mcq_results").select("*").order("created_at", desc=True).limit(10).execute()
        return jsonify({'success': True, 'history': response.data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@mcq_bp.route('/api/admin/backfill-usernames', methods=['POST'])
@admin_required
def backfill_usernames():
    try:
        supabase.table("mcq_results").update({"username": "unknown"}).is_("username", "null").execute()
        supabase.table("mcq_results").update({"username": "unknown"}).eq("username", "").execute()
        return jsonify({'success': True, 'message': 'Backfill complete'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@mcq_bp.route('/api/debug-role', methods=['GET'])
def debug_role():
    return jsonify({
        'raw_predicted_role': session.get('raw_predicted_role', 'Not set'),
        'normalized_role':    session.get('predicted_job_role', 'Not set'),
        'confidence':         session.get('prediction_confidence', 'Not set')
    })


@mcq_bp.route('/mcq_test')
def mcq_test():
    return render_template('mcq_test.html')

@mcq_bp.route('/api/reset-seen-questions', methods=['POST'])
def reset_seen_questions():
    session.pop('seen_question_ids', None)
    return jsonify({'success': True, 'message': 'Question history cleared'})