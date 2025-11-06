import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta


class AuthDatabase:
    """Handles all database operations for user authentication."""

    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self.init_database()

    # ---------------------------
    # Database Initialization
    # ---------------------------
    def init_database(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                phone_number TEXT,
                job_role TEXT,
                company_name TEXT,
                location TEXT,
                profile_pic TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # OTP table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS otp_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                otp_code TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                is_used INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()

    # ---------------------------
    # Helper Functions
    # ---------------------------
    def hash_password(self, password):
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()

    # ---------------------------
    # User Management
    # ---------------------------
    def create_user(self, name, email, password, phone_number='', job_role='',
                    company_name='', location='', profile_pic_path=None):
        """Create a new user account."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            hashed_password = self.hash_password(password)

            cursor.execute('''
                INSERT INTO users 
                (name, email, password, phone_number, job_role, company_name, location, profile_pic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, email, hashed_password, phone_number, job_role, company_name, location, profile_pic_path))

            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError:
            return False, "Email already exists!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def verify_user(self, email, password):
        """Verify user credentials (login)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        hashed_password = self.hash_password(password)

        cursor.execute('''
            SELECT id, name, email FROM users 
            WHERE email = ? AND password = ?
        ''', (email, hashed_password))

        user = cursor.fetchone()
        conn.close()

        if user:
            return True, {"id": user[0], "name": user[1], "email": user[2]}
        return False, None

    def get_user_details(self, email):
        """Fetch complete user details from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT name, email, phone_number, job_role, company_name, location, profile_pic
            FROM users WHERE email = ?
        ''', (email,))

        user = cursor.fetchone()
        conn.close()

        if user:
            return {
                "name": user[0],
                "email": user[1],
                "phone_number": user[2] or '',
                "job_role": user[3] or '',
                "company_name": user[4] or '',
                "location": user[5] or '',
                "profile_pic": user[6] or ''
            }
        return None

    def update_user_profile(self, email, name, phone_number, job_role,
                            company_name, location, profile_pic_path=None):
        """Update user profile information, including optional profile picture."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if profile_pic_path:
                cursor.execute('''
                    UPDATE users 
                    SET name = ?, phone_number = ?, job_role = ?, company_name = ?, 
                        location = ?, profile_pic = ?
                    WHERE email = ?
                ''', (name, phone_number, job_role, company_name, location, profile_pic_path, email))
            else:
                cursor.execute('''
                    UPDATE users 
                    SET name = ?, phone_number = ?, job_role = ?, company_name = ?, location = ?
                    WHERE email = ?
                ''', (name, phone_number, job_role, company_name, location, email))

            conn.commit()
            conn.close()
            return True, "Profile updated successfully!"
        except Exception as e:
            return False, f"Error: {str(e)}"

    # ---------------------------
    # OTP Management
    # ---------------------------
    def generate_otp(self, email):
        """Generate a 6-digit OTP for email verification."""
        otp = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        expires_at = datetime.now() + timedelta(minutes=10)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO otp_codes (email, otp_code, expires_at)
            VALUES (?, ?, ?)
        ''', (email, otp, expires_at))

        conn.commit()
        conn.close()

        return otp

    def verify_otp(self, email, otp):
        """Verify an OTP code for a given email."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id FROM otp_codes 
            WHERE email = ? AND otp_code = ? AND is_used = 0 AND expires_at > ?
            ORDER BY created_at DESC LIMIT 1
        ''', (email, otp, datetime.now()))

        result = cursor.fetchone()

        if result:
            cursor.execute('UPDATE otp_codes SET is_used = 1 WHERE id = ?', (result[0],))
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False
