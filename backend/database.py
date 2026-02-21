"""
database.py — MS SQL Server database handler for Resume Analyzer MCQ System
Place this file in the same folder as app.py
"""

import pyodbc
import json
import random
from typing import List, Dict, Optional


class Database:
    def __init__(self, server: str = 'localhost\\SQLEXPRESS', use_windows_auth: bool = True,
                 username: str = None, password: str = None):
        """
        Initialize database connection.

        Args:
            server:           SQL Server instance, e.g. 'localhost\\SQLEXPRESS'
            use_windows_auth: True = Windows Auth (recommended), False = SQL Auth
            username:         SQL Auth username (only if use_windows_auth=False)
            password:         SQL Auth password (only if use_windows_auth=False)
        """
        self.server = server
        self.database = 'ResumeAnalyzerDB'
        self.use_windows_auth = use_windows_auth
        self.username = username
        self.password = password
        self.connection = None

        # Connect and verify tables exist
        self._connect()
        self._ensure_tables_exist()

    # ─────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────

    def _get_connection_string(self) -> str:
        # Find the best available ODBC driver
        available_drivers = pyodbc.drivers()
        sql_drivers = [d for d in available_drivers if 'SQL Server' in d]

        if not sql_drivers:
            raise Exception(
                "No SQL Server ODBC driver found.\n"
                "Install 'ODBC Driver 17 for SQL Server' from Microsoft."
            )

        # Prefer newer drivers
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
        """Establish connection to SQL Server."""
        try:
            conn_str = self._get_connection_string()
            self.connection = pyodbc.connect(conn_str, autocommit=True)
            print(f"[DB] Connected to {self.server}/{self.database} ✅")
        except Exception as e:
            print(f"[DB] Connection failed: {e}")
            raise

    def _get_cursor(self):
        """Return a fresh cursor, reconnecting if needed."""
        try:
            # Test if connection is still alive
            self.connection.cursor().execute("SELECT 1")
        except Exception:
            print("[DB] Reconnecting...")
            self._connect()
        return self.connection.cursor()

    # ─────────────────────────────────────────────
    # TABLE SETUP
    # ─────────────────────────────────────────────

    def _ensure_tables_exist(self):
        """Create tables if they don't already exist."""
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
                SELECT * FROM sysobjects WHERE name='test_results' AND xtype='U'
            )
            CREATE TABLE test_results (
                id               INT PRIMARY KEY IDENTITY(1,1),
                job_role         NVARCHAR(100) NOT NULL,
                total_questions  INT NOT NULL,
                correct_answers  INT NOT NULL,
                score_percentage FLOAT NOT NULL,
                timestamp        DATETIME DEFAULT GETDATE()
            )
        """)

        print("[DB] Tables verified ✅")

    # ─────────────────────────────────────────────
    # HELPER: Parse options JSON from DB
    # ─────────────────────────────────────────────

    def _parse_options(self, options_raw) -> List[str]:
        """Parse options whether stored as JSON string or plain text."""
        if isinstance(options_raw, list):
            return options_raw
        if isinstance(options_raw, str):
            try:
                parsed = json.loads(options_raw)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # Fallback: comma-separated
            return [o.strip() for o in options_raw.split(',') if o.strip()]
        return []

    def _row_to_dict(self, row, columns) -> Dict:
        """Convert a pyodbc Row to a dict."""
        d = dict(zip(columns, row))
        if 'options' in d:
            d['options'] = self._parse_options(d['options'])
        return d

    # ─────────────────────────────────────────────
    # CORE: GET QUESTIONS
    # ─────────────────────────────────────────────

    def get_questions_by_role(self, job_role: str, limit: int = 10) -> List[Dict]:
        """
        Fetch randomized questions for a job role.

        Matching strategy (in order):
        1. Exact match (case-insensitive)
        2. Partial match (role contains or is contained by DB value)
        3. DEFAULT questions
        """
        try:
            cursor = self._get_cursor()

            # ── Strategy 1: Exact case-insensitive match ──
            cursor.execute("""
                SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                FROM mcq_questions
                WHERE UPPER(job_role) = UPPER(?)
            """, (job_role,))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            if not rows:
                # ── Strategy 2: Partial match ──
                print(f"[DB] Exact match failed for '{job_role}', trying partial match...")
                cursor.execute("""
                    SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM mcq_questions
                    WHERE UPPER(job_role) LIKE UPPER(?)
                       OR UPPER(?) LIKE UPPER(CONCAT('%', job_role, '%'))
                """, (f'%{job_role}%', job_role))
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

            if not rows:
                # ── Strategy 3: DEFAULT fallback ──
                print(f"[DB] No match for '{job_role}', loading DEFAULT questions...")
                cursor.execute("""
                    SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                    FROM mcq_questions
                    WHERE UPPER(job_role) = 'DEFAULT'
                """)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

            if not rows:
                print(f"[DB] ⚠️ No questions found at all for role: '{job_role}'")
                return []

            questions = [self._row_to_dict(row, columns) for row in rows]

            # Shuffle and limit
            random.shuffle(questions)
            result = questions[:limit]

            print(f"[DB] Returning {len(result)} questions for role '{job_role}' "
                  f"(from {len(questions)} available)")
            return result

        except Exception as e:
            print(f"[DB] get_questions_by_role error: {e}")
            raise

    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """Fetch a single question by its ID (used when evaluating answers)."""
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT id, job_role, question, options, correct_answer, difficulty, explanation
                FROM mcq_questions
                WHERE id = ?
            """, (question_id,))
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row, columns)
            return None
        except Exception as e:
            print(f"[DB] get_question_by_id error: {e}")
            raise

    # ─────────────────────────────────────────────
    # CORE: SAVE / RETRIEVE TEST RESULTS
    # ─────────────────────────────────────────────

    def save_test_result(self, result: Dict) -> bool:
        """Save a completed test result to the database."""
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                INSERT INTO test_results (job_role, total_questions, correct_answers, score_percentage)
                VALUES (?, ?, ?, ?)
            """, (
                result.get('job_role', 'Unknown'),
                result.get('total_questions', 0),
                result.get('correct_answers', 0),
                result.get('score_percentage', 0.0)
            ))
            print(f"[DB] Test result saved: {result.get('job_role')} — {result.get('score_percentage')}%")
            return True
        except Exception as e:
            print(f"[DB] save_test_result error: {e}")
            raise

    def get_test_history(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent test results."""
        try:
            cursor = self._get_cursor()
            cursor.execute("""
                SELECT TOP (?) id, job_role, total_questions, correct_answers,
                              score_percentage, timestamp
                FROM test_results
                ORDER BY timestamp DESC
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            history = []
            for row in rows:
                d = dict(zip(columns, row))
                # Make timestamp JSON-serializable
                if d.get('timestamp'):
                    d['timestamp'] = str(d['timestamp'])
                history.append(d)
            return history
        except Exception as e:
            print(f"[DB] get_test_history error: {e}")
            raise

    # ─────────────────────────────────────────────
    # ADMIN: ADD QUESTIONS
    # ─────────────────────────────────────────────

    def add_question(self, job_role: str, question: str, options: List[str],
                     correct_answer: str, difficulty: str = 'medium',
                     explanation: str = '') -> bool:
        """Add a new MCQ question to the database."""
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

    # ─────────────────────────────────────────────
    # DEBUG HELPERS
    # ─────────────────────────────────────────────

    def list_all_roles(self) -> List[Dict]:
        """Show all distinct job roles and their question counts — useful for debugging."""
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
        """Return total number of questions in the database."""
        try:
            cursor = self._get_cursor()
            cursor.execute("SELECT COUNT(*) FROM mcq_questions")
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"[DB] get_total_question_count error: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("[DB] Connection closed.")