# migrate_to_supabase.py — run ONCE
import pyodbc
import json
from supabase import create_client

SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=ResumeAnalyzerDB;"
    "Trusted_Connection=yes;"
)
cursor = conn.cursor()

print("Migrating mcq_questions → Supabase...")
cursor.execute("SELECT job_role, question, options, correct_answer, difficulty, explanation FROM mcq_questions")
rows = cursor.fetchall()

batch = []
for row in rows:
    job_role, question, options_raw, correct_answer, difficulty, explanation = row
    try:
        options = json.loads(options_raw) if options_raw else []
    except Exception:
        options = [o.strip() for o in options_raw.split(',') if o.strip()]

    batch.append({
        "job_role": job_role,
        "question": question,
        "options": options,
        "correct_answer": correct_answer,
        "difficulty": difficulty or "medium",
        "explanation": explanation or "",
        "status": "active"
    })

# Insert in chunks of 500
for i in range(0, len(batch), 500):
    chunk = batch[i:i+500]
    supabase.table("mcq_questions").insert(chunk).execute()
    print(f"  ✅ Inserted {len(chunk)} rows (batch {i//500 + 1})")

print(f"\n🎉 Done! {len(batch)} questions migrated.")
conn.close()