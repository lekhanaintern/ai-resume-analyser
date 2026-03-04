"""
database.py — Supabase cloud handler
Only handles mcq_questions table
"""

import random
from typing import List, Dict, Optional
from supabase import create_client, Client

SUPABASE_URL = "https://kxqzncqubkzxdjqkmstq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt4cXpuY3F1Ymt6eGRqcWttc3RxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE2NDg3MDksImV4cCI6MjA4NzIyNDcwOX0.82b2UA_f3BPpwwvymmcXqBiBxDCvC1EYf7nvryUefPI"


class Database:
    def __init__(self, **kwargs):  # kwargs absorbs any old MSSQL params
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[DB] Connected to Supabase ✅")

    def get_questions_by_role(
        self,
        job_role: str,
        limit: int = 10,
        exclude_ids: List[int] = None
    ) -> List[Dict]:
        exclude_ids = exclude_ids or []

        for strategy in ['exact', 'partial', 'default']:
            rows = self._fetch_questions(job_role, strategy, limit * 3, exclude_ids)
            if rows:
                random.shuffle(rows)
                return rows[:limit]

        print(f"[DB] ⚠️ No questions found for role: '{job_role}'")
        return []

    def _fetch_questions(
        self, job_role: str, strategy: str,
        limit: int, exclude_ids: List[int]
    ) -> List[Dict]:
        try:
            query = self.supabase.table("mcq_questions").select(
                "id, job_role, question, options, correct_answer, difficulty, explanation"
            ).eq("status", "active")

            if strategy == 'exact':
                query = query.ilike("job_role", job_role)
            elif strategy == 'partial':
                query = query.ilike("job_role", f"%{job_role}%")
            else:
                query = query.ilike("job_role", "DEFAULT")

            result = query.limit(limit).execute()
            rows = result.data or []

            if exclude_ids:
                rows = [r for r in rows if r['id'] not in exclude_ids]

            return rows
        except Exception as e:
            print(f"[DB] _fetch_questions error: {e}")
            return []

    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        try:
            result = self.supabase.table("mcq_questions").select(
                "id, job_role, question, options, correct_answer, difficulty, explanation"
            ).eq("id", question_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"[DB] get_question_by_id error: {e}")
            return None

    def set_question_status(self, question_id: int, status: str) -> bool:
        try:
            self.supabase.table("mcq_questions").update({"status": status}).eq("id", question_id).execute()
            return True
        except Exception as e:
            print(f"[DB] set_question_status error: {e}")
            return False

    def add_question(self, job_role: str, question: str, options: List[str],
                     correct_answer: str, difficulty: str = 'medium',
                     explanation: str = '') -> bool:
        try:
            self.supabase.table("mcq_questions").insert({
                "job_role": job_role, "question": question,
                "options": options, "correct_answer": correct_answer,
                "difficulty": difficulty, "explanation": explanation,
                "status": "active"
            }).execute()
            return True
        except Exception as e:
            print(f"[DB] add_question error: {e}")
            return False

    def list_all_roles(self) -> List[Dict]:
        try:
            result = self.supabase.table("mcq_questions").select("job_role").execute()
            role_counts = {}
            for row in result.data:
                role = row['job_role']
                role_counts[role] = role_counts.get(role, 0) + 1
            return [{'job_role': r, 'question_count': c} for r, c in sorted(role_counts.items())]
        except Exception as e:
            print(f"[DB] list_all_roles error: {e}")
            return []

    def get_total_question_count(self) -> int:
        try:
            result = self.supabase.table("mcq_questions").select("id", count='exact').execute()
            return result.count or 0
        # ✅ Actually use `e` in the print
        except Exception as e:
            print(f"[DB] get_total_question_count error: {e}")
            return 0

    def close(self):
        pass