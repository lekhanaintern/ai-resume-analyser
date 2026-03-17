"""
merge_datasets.py
=================
Merges all 3 resume datasets into one merged_resumes.csv
Run from: backend/data/ folder
"""

import pandas as pd

# ── Load all 3 datasets ───────────────────────────────────────────
print("Loading datasets...")

df1 = pd.read_csv('resumes_clean.csv')
df2 = pd.read_csv('resumesx.csv')
df3 = pd.read_csv('github_resumes.csv', encoding='latin-1', on_bad_lines='skip')

print(f"Dataset 1 (resumes_clean) : {len(df1)} resumes")
print(f"Dataset 2 (resumesx)      : {len(df2)} resumes")
print(f"Dataset 3 (github)        : {len(df3)} resumes")

# ── Standardise column names ──────────────────────────────────────
df1 = df1.rename(columns={'Resume': 'Resume', 'Category': 'Category'})[['Resume', 'Category']]
df2 = df2.rename(columns={'Resume': 'Resume', 'Category': 'Category'})[['Resume', 'Category']]
df3 = df3.rename(columns={'Text': 'Resume',   'Category': 'Category'})[['Resume', 'Category']]

# ── Normalise category names → UPPER-HYPHEN ──────────────────────
def normalise(cat):
    return str(cat).strip().upper().replace(' ', '-').replace('_', '-')

df1['Category'] = df1['Category'].apply(normalise)
df2['Category'] = df2['Category'].apply(normalise)
df3['Category'] = df3['Category'].apply(normalise)

# ── Role name mapping ─────────────────────────────────────────────
ROLE_MAP = {
    'HUMAN-RESOURCES'          : 'HR',
    'BUILDING-AND-CONSTRUCTION': 'CONSTRUCTION',
    'HEALTH-AND-FITNESS'       : 'FITNESS',
    'EDUCATION'                : 'TEACHER',
    'MANAGEMENT'               : 'BUSINESS-DEVELOPMENT',
    'ARCHITECTURE'             : 'DESIGNER',
    'SQL-DEVELOPER'            : 'DATA-ANALYST',
    'DATA-SCIENCE'             : 'DATA-SCIENCE',
    # Fix duplicates — merge into one standard name
    'DESIGNING'                : 'DESIGNER',
    'DEVOPS-ENGINEER'          : 'DEVOPS',
    'FOOD-AND-BEVERAGES'       : 'CHEF',
    'HADOOP'                   : 'DATABASE',
    'AUTOMATION-TESTING'       : 'TESTING',
}

df3['Category'] = df3['Category'].replace(ROLE_MAP)

# ── Merge all 3 ───────────────────────────────────────────────────
merged = pd.concat([df1, df2, df3], ignore_index=True)
merged = merged.dropna(subset=['Resume', 'Category'])
merged = merged[merged['Resume'].str.strip().str.len() > 50]
merged = merged.drop_duplicates(subset=['Resume'])

# ── Remove roles with fewer than 50 samples ───────────────────────
counts = merged['Category'].value_counts()
valid_roles = counts[counts >= 50].index
removed = counts[counts < 50]
if len(removed) > 0:
    print(f"\n🗑️  Removing {len(removed)} roles with < 50 samples:")
    print(removed.to_string())
merged = merged[merged['Category'].isin(valid_roles)]

# ── Show final distribution ───────────────────────────────────────
final_counts = merged['Category'].value_counts()
print(f"\n{'='*55}")
print(f"FINAL: {len(merged)} resumes, {merged['Category'].nunique()} categories")
print(f"{'='*55}")
print(final_counts.to_string())

# ── Save ──────────────────────────────────────────────────────────
merged.to_csv('merged_resumes.csv', index=False)
print("\n✅ Saved to merged_resumes.csv")
print("Now run:")
print("  cd ../models")
print("  python train_model.py --data ../data/merged_resumes.csv")