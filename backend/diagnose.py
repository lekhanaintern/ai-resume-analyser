"""
diagnose.py — Run this FIRST to verify your setup is working correctly.
Place in the same folder as app.py and run: python diagnose.py
"""

import sys
sys.path.insert(0, '.')

print("\n" + "=" * 60)
print("  RESUME ANALYZER — MCQ DIAGNOSTIC TOOL")
print("=" * 60)

# ── Step 1: Check pyodbc ──────────────────────────────────────
print("\n[1] Checking pyodbc...")
try:
    import pyodbc
    drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d]
    print(f"    ✅ pyodbc installed. SQL drivers found: {drivers}")
except ImportError:
    print("    ❌ pyodbc not installed. Run: pip install pyodbc")
    sys.exit(1)

# ── Step 2: Connect to database ──────────────────────────────
print("\n[2] Connecting to database...")
try:
    from database import Database
    db = Database(server='localhost\\SQLEXPRESS', use_windows_auth=True)
    print("    ✅ Connected successfully")
except Exception as e:
    print(f"    ❌ Connection failed: {e}")
    sys.exit(1)

# ── Step 3: List all roles in DB ─────────────────────────────
print("\n[3] Roles found in mcq_questions table:")
try:
    roles = db.list_all_roles()
    total = db.get_total_question_count()
    if roles:
        for r in roles:
            print(f"    • {r['job_role']:<30} → {r['question_count']} questions")
        print(f"\n    Total questions in DB: {total}")
    else:
        print("    ⚠️  No questions found! Did you run the SQL script in SSMS?")
except Exception as e:
    print(f"    ❌ Error: {e}")
    sys.exit(1)

# ── Step 4: Test role normalization ──────────────────────────
print("\n[4] Testing role normalization (simulating ML model output):")

test_roles = [
    'Data Scientist',
    'data scientist',
    'DATA-SCIENTIST',
    'Web Developer',
    'WEB DEVELOPER',
    'web-developer',
    'Python Developer',
    'HR',
    'human resources',
    'INFORMATION-TECHNOLOGY',
    'Information Technology',
    'Designer',
    'DESIGNER',
    'Finance',
    'Banking',
    'Unknown Role XYZ',
]

# Inline normalization map for this test
ROLE_MAP = {
    'data scientist': 'DATA-SCIENCE', 'data science': 'DATA-SCIENCE',
    'data-science': 'DATA-SCIENCE', 'data-scientist': 'DATA-SCIENCE',
    'web developer': 'WEB-DEVELOPER', 'web-developer': 'WEB-DEVELOPER',
    'web development': 'WEB-DEVELOPER', 'webdeveloper': 'WEB-DEVELOPER',
    'python developer': 'DATA-SCIENCE', 'python': 'DATA-SCIENCE',
    'hr': 'HR', 'human resources': 'HR', 'human resource': 'HR',
    'designer': 'DESIGNER', 'ui designer': 'DESIGNER', 'ux designer': 'DESIGNER',
    'information technology': 'INFORMATION-TECHNOLOGY',
    'information-technology': 'INFORMATION-TECHNOLOGY', 'it': 'INFORMATION-TECHNOLOGY',
    'teacher': 'TEACHER', 'advocate': 'ADVOCATE', 'lawyer': 'ADVOCATE',
    'business development': 'BUSINESS-DEVELOPMENT', 'bd': 'BUSINESS-DEVELOPMENT',
    'healthcare': 'HEALTHCARE', 'medical': 'HEALTHCARE', 'doctor': 'HEALTHCARE',
    'fitness': 'FITNESS', 'agriculture': 'AGRICULTURE', 'bpo': 'BPO',
    'sales': 'SALES', 'consultant': 'CONSULTANT', 'consulting': 'CONSULTANT',
    'digital media': 'DIGITAL-MEDIA', 'digital marketing': 'DIGITAL-MEDIA',
    'automobile': 'AUTOMOBILE', 'automotive': 'AUTOMOBILE',
    'chef': 'CHEF', 'cook': 'CHEF',
    'finance': 'FINANCE', 'financial analyst': 'FINANCE',
    'apparel': 'APPAREL', 'fashion': 'APPAREL',
    'engineering': 'ENGINEERING', 'engineer': 'ENGINEERING',
    'accountant': 'ACCOUNTANT', 'accounting': 'ACCOUNTANT',
    'construction': 'CONSTRUCTION', 'public relations': 'PUBLIC-RELATIONS',
    'pr': 'PUBLIC-RELATIONS', 'banking': 'BANKING', 'bank': 'BANKING',
    'arts': 'ARTS', 'artist': 'ARTS', 'aviation': 'AVIATION', 'pilot': 'AVIATION',
    'general': 'DEFAULT', 'default': 'DEFAULT',
}

VALID_ROLES = {
    'HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
    'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE', 'BPO',
    'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE', 'CHEF', 'FINANCE',
    'APPAREL', 'ENGINEERING', 'ACCOUNTANT', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'BANKING', 'ARTS', 'AVIATION', 'DATA-SCIENCE', 'WEB-DEVELOPER', 'DEFAULT'
}

def normalize(role):
    if role in VALID_ROLES:
        return role
    lower = role.lower().strip().replace('_', '-')
    if lower in ROLE_MAP:
        return ROLE_MAP[lower]
    for key, val in ROLE_MAP.items():
        if key in lower or lower in key:
            return val
    return 'DEFAULT'

for role in test_roles:
    normalized = normalize(role)
    print(f"    '{role:<30}' → '{normalized}'")

# ── Step 5: Test DB query for each mapped role ────────────────
print("\n[5] Testing DB queries for key roles:")
key_roles = ['DATA-SCIENCE', 'WEB-DEVELOPER', 'HR', 'DESIGNER',
             'INFORMATION-TECHNOLOGY', 'FINANCE', 'BANKING', 'DEFAULT']

for role in key_roles:
    try:
        questions = db.get_questions_by_role(role, limit=3)
        status = f"✅ {len(questions)} questions found" if questions else "⚠️  0 questions found"
        print(f"    {role:<30} → {status}")
    except Exception as e:
        print(f"    {role:<30} → ❌ Error: {e}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNOSIS COMPLETE")
print("=" * 60)
print("""
Next steps:
  • If all roles show ✅ questions found → your fix is working!
  • If some roles show ⚠️  0 questions → re-run the SQL script in SSMS
  • Copy fixed_app.py  → replace your existing app.py
  • Copy fixed_database.py → replace your existing database.py
  • Restart Flask: python app.py
  • Visit http://localhost:5000/api/debug-role after uploading a resume
    to confirm the role is being detected correctly
""")