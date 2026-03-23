import os
import time as _time
import socket
from supabase import create_client, Client
from dotenv import load_dotenv

# ── Force IPv4 — fixes WinError 10060 on IPv6 networks (Windows) ───────────
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo
# ────────────────────────────────────────────────────────────────────────────

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing required environment variables: SUPABASE_URL and SUPABASE_KEY. "
        "Copy .env.example to .env and fill in the values."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Simple in-memory cache ──────────────────────────────────────────────────
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