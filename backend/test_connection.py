from supabase import create_client

SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table("mcq_questions").select("id").limit(1).execute()
    print("✅ Connected! Table exists.")
except Exception as e:
    print(f"❌ Connection failed: {e}")