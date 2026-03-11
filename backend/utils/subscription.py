from config.settings import supabase


def _get_free_plan_id():
    """Fetch the UUID of the Free plan from subscription_plans."""
    try:
        r = supabase.table("subscription_plans").select("id").eq("name", "Free").limit(1).execute()
        if r.data:
            return r.data[0]['id']
    except Exception as e:
        print(f"[_get_free_plan_id] ERROR: {e}")
    return None


def _ensure_subscription_row(username):
    try:
        r = supabase.table("user_subscriptions") \
            .select("*, subscription_plans(*)") \
            .eq("username", username) \
            .eq("status", "active") \
            .order("created_at", desc=True) \
            .limit(1).execute()
        if r.data:
            return r.data[0]

        free_id = _get_free_plan_id()
        if not free_id:
            return None

        supabase.table("user_subscriptions").insert({
            "username":     username,
            "plan_id":      free_id,
            "status":       "active",
            "resumes_used": 0,
            "mcq_used":     0,
        }).execute()

        r2 = supabase.table("user_subscriptions") \
            .select("*, subscription_plans(*)") \
            .eq("username", username) \
            .eq("status", "active") \
            .order("created_at", desc=True) \
            .limit(1).execute()
        return r2.data[0] if r2.data else None

    except Exception as e:
        print(f"[_ensure_subscription_row] ERROR: {e}")
        return None


def get_user_subscription(username):
    return _ensure_subscription_row(username)


def check_limit(username, action):
    try:
        sub = _ensure_subscription_row(username)
        if not sub:
            print(f"[check_limit] WARNING: Could not load subscription for {username}. Allowing.")
            return True, ""

        plan  = sub.get("subscription_plans") or {}
        if action == "resume":
            limit = plan.get("max_resumes", 2)
            used  = sub.get("resumes_used", 0)
            label = "resume analyses"
        else:
            limit = plan.get("max_mcq_tests", 1)
            used  = sub.get("mcq_used", 0)
            label = "MCQ tests"

        if limit == -1:
            return True, ""

        print(f"[check_limit] user={username} action={action} used={used} limit={limit}")

        if used >= limit:
            plan_name = plan.get("name", "Free")
            return False, (
                f"You've used all {limit} {label} on your {plan_name} plan. "
                f"Contact your admin to upgrade."
            )
        return True, ""

    except Exception as e:
        print(f"[check_limit] ERROR: {e} — allowing as fallback")
        return True, ""


def increment_usage(username, action):
    try:
        r = supabase.table("user_subscriptions") \
            .select("id, resumes_used, mcq_used") \
            .eq("username", username) \
            .eq("status", "active") \
            .order("created_at", desc=True) \
            .limit(1).execute()

        if not r.data:
            print(f"[increment_usage] No active subscription row for {username}, skipping.")
            return

        row     = r.data[0]
        field   = "resumes_used" if action == "resume" else "mcq_used"
        new_val = (row.get(field) or 0) + 1

        supabase.table("user_subscriptions") \
            .update({field: new_val}) \
            .eq("id", row["id"]).execute()

        print(f"[increment_usage] user={username} action={action} new_{field}={new_val}")

    except Exception as e:
        print(f"[increment_usage] ERROR: {e}")