import time as _time
from supabase import create_client, Client

SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"

# Anon key — normal DB operations
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"

# Service-role key — admin auth calls only (backend only, never expose to frontend)
# GET IT: Supabase Dashboard → Project Settings → API → service_role → Copy
SUPABASE_SERVICE_KEY = "PASTE_YOUR_SERVICE_ROLE_KEY_HERE"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

_stats_cache: dict = {}

def _cache_get(key: str, ttl: int = 300):
    entry = _stats_cache.get(key)
    if entry and (_time.time() - entry['ts']) < ttl:
        return entry['val']
    return None

def _cache_set(key: str, val) -> None:
    _stats_cache[key] = {'val': val, 'ts': _time.time()}

def _cache_clear() -> None:
    _stats_cache.clear()