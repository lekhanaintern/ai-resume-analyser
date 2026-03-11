from flask import Blueprint, request, jsonify, session
from config.settings import supabase, _cache_clear
from utils.auth import admin_required
from utils.subscription import get_user_subscription

subscriptions_bp = Blueprint('subscriptions', __name__)


# ============================================================
# SUBSCRIPTION ROUTES — Candidate
# ============================================================
@subscriptions_bp.route('/api/my-subscription', methods=['GET'])
def my_subscription():
    username = session.get('user_username')
    if not username:
        return jsonify({'error': 'Not logged in'}), 401
    sub = get_user_subscription(username)
    return jsonify({'success': True, 'subscription': sub})


# ============================================================
# SUBSCRIPTION ROUTES — Admin
# ============================================================
@subscriptions_bp.route('/api/plans', methods=['GET'])
def get_plans_public():
    try:
        plans = supabase.table("subscription_plans").select("*").eq("is_active", True).order("price_monthly").execute()
        return jsonify({'success': True, 'plans': plans.data or []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/plans', methods=['GET'])
@admin_required
def admin_get_plans():
    try:
        plans = supabase.table("subscription_plans").select("*").eq("is_active", True).execute()
        return jsonify({'success': True, 'plans': plans.data or []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# UPDATE PLAN PRICE & LIMITS
# POST /api/admin/update-plan
# Body: { id, name, price_monthly, price_yearly, max_resumes, max_mcq_tests }
# ============================================================
@subscriptions_bp.route('/api/admin/update-plan', methods=['POST'])
@admin_required
def admin_update_plan():
    try:
        data = request.get_json(force=True, silent=True) or {}

        plan_id       = data.get('id')
        name          = (data.get('name') or '').strip()
        price_monthly = data.get('price_monthly')
        price_yearly  = data.get('price_yearly')
        max_resumes   = data.get('max_resumes')
        max_mcq_tests = data.get('max_mcq_tests')

        # Validate
        if not plan_id:
            return jsonify({'error': 'Plan ID is required'}), 400
        if not name:
            return jsonify({'error': 'Plan name is required'}), 400
        if price_monthly is None or int(price_monthly) < 0:
            return jsonify({'error': 'price_monthly must be 0 or greater'}), 400
        if price_yearly is None or int(price_yearly) < 0:
            return jsonify({'error': 'price_yearly must be 0 or greater'}), 400
        if max_resumes is None or int(max_resumes) < -1:
            return jsonify({'error': 'max_resumes must be -1 (unlimited) or greater'}), 400
        if max_mcq_tests is None or int(max_mcq_tests) < -1:
            return jsonify({'error': 'max_mcq_tests must be -1 (unlimited) or greater'}), 400

        # Check plan exists
        existing = supabase.table("subscription_plans").select("id").eq("id", plan_id).execute()
        if not existing.data:
            return jsonify({'error': 'Plan not found'}), 404

        # Update
        supabase.table("subscription_plans").update({
            "name":          name,
            "price_monthly": int(price_monthly),
            "price_yearly":  int(price_yearly),
            "max_resumes":   int(max_resumes),
            "max_mcq_tests": int(max_mcq_tests),
        }).eq("id", plan_id).execute()

        _cache_clear()  # Invalidate stats cache

        print(f"[update-plan] Plan '{name}' (id={plan_id}) updated by admin '{session.get('user_username')}'")
        return jsonify({'success': True})

    except Exception as e:
        print(f"[update-plan] ERROR: {e}")
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/subscriptions', methods=['GET'])
@admin_required
def admin_get_subscriptions():
    try:
        subs = supabase.table("user_subscriptions") \
            .select("*, subscription_plans(name, price_monthly, price_yearly, max_resumes, max_mcq_tests)") \
            .order("created_at", desc=True).execute()
        return jsonify({'success': True, 'subscriptions': subs.data or []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/assign-plan', methods=['POST'])
@admin_required
def admin_assign_plan():
    try:
        data     = request.get_json()
        username = data.get('username')
        plan_id  = data.get('plan_id')
        expires  = data.get('expires_at') or None
        if not username or not plan_id:
            return jsonify({'error': 'username and plan_id required'}), 400
        supabase.table("user_subscriptions") \
            .update({"status": "cancelled"}) \
            .eq("username", username).eq("status", "active").execute()
        supabase.table("user_subscriptions").insert({
            "username":    username,
            "plan_id":     plan_id,
            "status":      "active",
            "expires_at":  expires,
            "resumes_used": 0,
            "mcq_used":    0
        }).execute()
        _cache_clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/revoke-subscription', methods=['POST'])
@admin_required
def admin_revoke_subscription():
    try:
        username = request.get_json().get('username')
        supabase.table("user_subscriptions") \
            .update({"status": "cancelled"}) \
            .eq("username", username).eq("status", "active").execute()
        _cache_clear()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/reset-usage', methods=['POST'])
@admin_required
def admin_reset_usage():
    try:
        username = request.get_json().get('username')
        supabase.table("user_subscriptions") \
            .update({"resumes_used": 0, "mcq_used": 0}) \
            .eq("username", username).eq("status", "active").execute()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# PLAN REQUEST ROUTES
# ============================================================
@subscriptions_bp.route('/api/request-plan', methods=['POST'])
def request_plan():
    username = session.get('user_username')
    if not username:
        return jsonify({'error': 'Not logged in'}), 401
    try:
        data               = request.get_json()
        requested_plan_id  = data.get('plan_id')
        requested_plan_name= data.get('plan_name', '')
        billing_cycle      = data.get('billing_cycle', 'monthly')
        message            = (data.get('message') or '').strip()[:500]

        if not requested_plan_id:
            return jsonify({'error': 'plan_id required'}), 400

        existing = supabase.table("plan_requests") \
            .select("id") \
            .eq("username", username) \
            .eq("requested_plan_id", requested_plan_id) \
            .eq("status", "pending").execute()
        if existing.data:
            return jsonify({'error': 'You already have a pending request for this plan.'}), 400

        sub = get_user_subscription(username)
        current_plan = (sub or {}).get('subscription_plans', {}).get('name', 'Free')

        supabase.table("plan_requests").insert({
            "username":            username,
            "current_plan":        current_plan,
            "requested_plan_id":   requested_plan_id,
            "requested_plan_name": requested_plan_name,
            "status":              "pending",
            "message":             f"[{billing_cycle.upper()}] {message}".strip(),
        }).execute()

        cycle_label = "yearly" if billing_cycle == "yearly" else "monthly"
        return jsonify({'success': True, 'message': f'Request for {requested_plan_name} plan ({cycle_label}) submitted! Your admin will review it shortly.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/my-plan-requests', methods=['GET'])
def my_plan_requests():
    username = session.get('user_username')
    if not username:
        return jsonify({'error': 'Not logged in'}), 401
    try:
        r = supabase.table("plan_requests") \
            .select("*") \
            .eq("username", username) \
            .order("created_at", desc=True) \
            .limit(10).execute()
        return jsonify({'success': True, 'requests': r.data or []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/plan-requests', methods=['GET'])
@admin_required
def admin_get_plan_requests():
    try:
        status = request.args.get('status', '')
        q = supabase.table("plan_requests").select("*").order("created_at", desc=True)
        if status:
            q = q.eq("status", status)
        r = q.limit(100).execute()
        return jsonify({'success': True, 'requests': r.data or []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@subscriptions_bp.route('/api/admin/resolve-plan-request', methods=['POST'])
@admin_required
def admin_resolve_plan_request():
    try:
        data       = request.get_json()
        request_id = data.get('request_id')
        action     = data.get('action')
        admin_note = (data.get('admin_note') or '').strip()

        if action not in ('approve', 'reject'):
            return jsonify({'error': 'action must be approve or reject'}), 400

        req_r = supabase.table("plan_requests").select("*").eq("id", request_id).limit(1).execute()
        if not req_r.data:
            return jsonify({'error': 'Request not found'}), 404
        req = req_r.data[0]

        if req['status'] != 'pending':
            return jsonify({'error': 'Request already resolved'}), 400

        new_status = 'approved' if action == 'approve' else 'rejected'

        if action == 'approve':
            username = req['username']
            plan_id  = req['requested_plan_id']
            supabase.table("user_subscriptions") \
                .update({"status": "cancelled"}) \
                .eq("username", username).eq("status", "active").execute()
            supabase.table("user_subscriptions").insert({
                "username":     username,
                "plan_id":      plan_id,
                "status":       "active",
                "resumes_used": 0,
                "mcq_used":     0,
            }).execute()
            _cache_clear()

        from datetime import datetime as _dt
        supabase.table("plan_requests").update({
            "status":      new_status,
            "admin_note":  admin_note,
            "resolved_at": _dt.utcnow().isoformat(),
        }).eq("id", request_id).execute()

        return jsonify({'success': True, 'status': new_status})
    except Exception as e:
        return jsonify({'error': str(e)}), 500