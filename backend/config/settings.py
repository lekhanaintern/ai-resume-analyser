import time as _time
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"

# Anon key — normal DB operations
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"

# ── Email config for OTP sending (add these to your .env file) ──
# EMAIL_ADDRESS  = your Gmail address      e.g. yourapp@gmail.com
# EMAIL_PASSWORD = your Gmail App Password (NOT your real Gmail password)
#   How to get App Password:
#   Gmail → Settings → Security → 2-Step Verification → App Passwords → Generate

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