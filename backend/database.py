"""
database.py — MS SQL Server database handler
Only handles question bank (mcq_questions, interview_mcq_questions)
MCQ Results are now stored in Supabase
"""

import pyodbc
import json
from typing import List, Dict, Optional


class Database:
    def __init__(self, server: str = 'localhost\\SQLEXPRESS', use_windows_auth: bool = True,
                 username: str = None, password: str = None):
        self.server = server
        self.database = 'ResumeAnalyzerDB'
        self.use_windows_auth = use_windows_auth
        self.username = username
        self.password = password
        self.connection = None

        self._connect()
        self._ensure_tables_exist()

    # ─────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────

    def _get_connection_string(self) -> str:
        available_drivers = pyodbc.drivers()
        sql_drivers = [d for d in available_drivers if 'SQL Server' in d]

        if not sql_drivers:
            raise Exception(
                "No SQL Server ODBC driver found.\n"
                "Install 'ODBC Driver 17 for SQL Server' from Microsoft."
            )

        preferred = ['ODBC Driver 17 for SQL Server', 'ODBC Driver 13 for SQL Server',
                     'SQL Server Native Client 11.0', 'SQL Server']
        driver = sql_drivers[0]
        for p in preferred:
            if p in sql_drivers:
                driver = p
                break

        print(f"[DB] Using driver: {driver}")

        if self.use_windows_auth:
            return (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            return (
                f"DRIVER={{{driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )

    def _connect(self):
        try:
            conn_str = self._get_connection_string()
            self.connection = pyodbc.connect(conn_str, autocommit=True)
            print(f"[DB] Connected to {self.server}/{self.database} ✅")
        except Exception as e:
            print(f"[DB] Connection failed: {e}")
            raise

    def _get_cursor(self):
        try:
            self.connection.cursor().execute("SELECT 1")
        except Exception:
            print("[DB] Reconnecting...")
            self._connect()
        return self.connection.cursor()

    # ─────────────────────────────────────────────
    # TABLE SETUP — only question bank tables
    # ─────────────────────────────────────────────

    def _ensure_tables_exist(self):
        cursor = self._get_cursor()

        cursor.execute("""
            IF NOT EXISTS (
                SELECT * FROM sysobjects WHERE name='mcq_questions' AND xtype='U'
            )
            CREATE TABLE mcq_questions (
                id              INT PRIMARY KEY IDENTITY(1,1),
                job_role        NVARCHAR(100) NOT NULL,
                question        NVARCHAR(MAX) NOT NULL,
                options         NVARCHAR(MAX) NOT NULL,
                correct_answer  NVARCHAR(500) NOT NULL,
                difficulty      NVARCHAR(20) DEFAULT 'medium',
                explanation     NVARCHAR(MAX),
                created_at      DATETIME DEFAULT GETDATE()
            )
        """)

        cursor.execute("""
            IF NOT EXISTS (
                SELECT * FROM sysobjects WHERE name='interview_mcq_questions' AND xtype='U'
            )
            CREATE TABLE interview_mcq_questions (
                id              INT PRIMARY KEY IDENTITY(1,1),
                job_role        NVARCHAR(100) NOT NULL,
                question        NVARCHAR(MAX) NOT NULL,
                options         NVARCHAR(MAX) NOT NULL,
                correct_answer  NVARCHAR(500) NOT NULL,
                difficulty      NVARCHAR(20) DEFAULT 'medium',
                explanation     NVARCHAR(MAX),
                created_at      DATETIME DEFAULT GETDATE()
            )
        """)

        print("[DB] Tables verified ✅")

    # ─────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────

    def _parse_options(self, options_raw) -> List[str]:
        if isinstance(options_raw, list):
            return options_raw
        if isinstance(options_raw, str):
            try:
                parsed = json.loads(options_raw)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            return [o.strip() for o in options_raw.split(',') if o.strip()]
        return []

    def _row_to_dict(self, row, columns) -> Dict:
        d = dict(zip(columns, row))
        if 'options' in d:
            d['options'] = self._parse_options(d['options'])
        return d

    # ─────────────────────────────────────────────
    # CORE: GET QUESTIONS — SQL-level randomization
    # ─────────────────────────────────────────────

    def get_questions_by_role(self, job_role: str, limit: int = 10) -> List[Dict]:
        """
        Fetch RANDOMLY selected questions for a job role using NEWID().
        Matching strategy:
        1. Exact match
        2. Partial match
        3. DEFAULT fallback
        """
        try:
            cursor = self._get_cursor()

            # Strategy 1: Exact match
            cursor.execute("""
                SELECT TOP (?) id, job_role, question, options, correct_answer, difficulty, explanation
                FROM mcq_questions
                WHERE UPPER(job_role) = UPPER(?)
                ORDER BY NEWID()
            """, (limit, job_role))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                # Strategy 2: Partial match
                print(f"[DB] Exact match failed for '{job_role}', trying partial match...")
                cursor.execute("""
                    SELECT TOP (?) id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM mcq_questions
                    WHERE UPPER(job_role) LIKE UPPER(?)
                       OR UPPER(?) LIKE UPPER(CONCAT('%', job_role, '%'))
                    ORDER BY NEWID()
                """, (limit, f'%{job_role}%', job_role))

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

            if not rows:
                # Strategy 3: DEFAULT fallback
                print(f"[DB] No match for '{job_role}', loading DEFAULT questions...")
                cursor.execute("""
                    SELECT TOP (?) id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM mcq_questions
                    WHERE UPPER(job_role) = 'DEFAULT'
                    ORDER BY NEWID()
                """, (limit,))

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

            if not rows:
                print(f"[DB] ⚠️ No questions found at all for role: '{job_role}'")
                return []

            result = [self._row_to_dict(row, columns) for row in rows]
            print(f"[DB] ✅ Returning {len(result)} random questions for role '{job_role}'")
            return result

        except Exception as e:
            print(f"[DB] get_questions_by_role error: {e}")
            raise

    # ─────────────────────────────────────────────
    # INTERVIEW MCQ — separate table, same logic
    # ─────────────────────────────────────────────

    def get_interview_questions_by_role(self, job_role: str, limit: int = 10) -> List[Dict]:
        try:
            cursor = self._get_cursor()

            cursor.execute("""
                SELECT TOP (?) id, job_role, question, options, correct_answer, difficulty, explanation
                FROM interview_mcq_questions
                WHERE UPPER(job_role) = UPPER(?)
                ORDER BY NEWID()
            """, (limit, job_role))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                cursor.execute("""
                    SELECT TOP (?) id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM interview_mcq_questions
                    WHERE UPPER(job_role) = 'DEFAULT'
                    ORDER BY NEWID()
                """, (limit,))
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

            if not rows:
                return []

            result = [self._row_to_dict(row, columns) for row in rows]
            print(f"[DB] ✅ Returning {len(result)} interview questions for role '{job_role}'")
            return result

        except Exception as e:
            print(f"[DB] get_interview_questions_by_role error: {e}")
            raise

    # ─────────────────────────────────────────────
    # GET QUESTION BY ID
    # ─────────────────────────────────────────────

    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """Fetch a single question by ID — checks both tables."""
        try:
            cursor = self._get_cursor()

            cursor.execute("""
                SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                FROM mcq_questions
                WHERE id = ?
            """, (question_id,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()

            if not row:
                cursor.execute("""
                    SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM interview_mcq_questions
                    WHERE id = ?
                """, (question_id,))
                columns = [desc[0] for desc in cursor.description]
                row = cursor.fetchone()

            return self._row_to_dict(row, columns) if row else None

        except Exception as e:
            print(f"[DB] get_question_by_id error: {e}")
            raise

    # ─────────────────────────────────────────────
    # ADMIN HELPERS
    # ─────────────────────────────────────────────

    def add_question(self, job_role: str, question: str, options: List[str],
                     correct_answer: str, difficulty: str = 'medium',
                     explanation: str = '') -> bool:
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                INSERT INTO mcq_questions (job_role, question, options, correct_answer, difficulty, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                job_role,
                question,
                json.dumps(options),
                correct_answer,
                difficulty,
                explanation
            ))
            return True
        except Exception as e:
            print(f"[DB] add_question error: {e}")
            raise

    def list_all_roles(self) -> List[Dict]:
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT job_role, COUNT(*) as question_count
                FROM mcq_questions
                GROUP BY job_role
                ORDER BY job_role
            """)
            rows = cursor.fetchall()
            return [{'job_role': row[0], 'question_count': row[1]} for row in rows]
        except Exception as e:
            print(f"[DB] list_all_roles error: {e}")
            raise

    def get_total_question_count(self) -> int:
        try:
            cursor = self._get_cursor()
            cursor.execute("SELECT COUNT(*) FROM mcq_questions")
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"[DB] get_total_question_count error: {e}")
            raise

    def close(self):
        if self.connection:
            self.connection.close()
            print("[DB] Connection closed.")