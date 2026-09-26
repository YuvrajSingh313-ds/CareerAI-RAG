import hashlib
import os
import secrets
import smtplib
import sqlite3
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, EmailStr

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "careerai_users.db"

load_dotenv(BASE_DIR / ".env")

OTP_EXPIRY_SECONDS = 10 * 60
OTP_RESEND_SECONDS = 30
MAX_OTP_ATTEMPTS = 5

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME)

app = FastAPI(
    title="CareerAI Authentication API",
    version="1.0.0"
)


# ---------------------------------------------------------
# DATABASE
# ---------------------------------------------------------

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_id TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL,
                last_login TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS otp_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                used INTEGER NOT NULL DEFAULT 0
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


init_db()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def hash_value(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def clean_email(value: str) -> str:
    return value.strip().lower()


def send_email_otp(email: str, otp: str, name: str):
    """
    Sends the real OTP to the email entered by the user.
    """

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        raise RuntimeError(
            "Email service is not configured. "
            "Please configure SMTP_USERNAME and SMTP_PASSWORD in .env."
        )

    message = EmailMessage()

    message["Subject"] = "CareerAI - Your Login OTP"
    message["From"] = SMTP_FROM
    message["To"] = email

    message.set_content(
        f"""Hello {name},

Your CareerAI verification code is:

{otp}

This OTP is valid for 10 minutes.

If you did not request this code, you can safely ignore this email.

Regards,
CareerAI
"""
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class OTPRequest(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100
    )

    email: EmailStr


class OTPVerify(BaseModel):
    name: str = Field(
        min_length=2,
        max_length=100
    )

    email: EmailStr

    otp: str = Field(
        pattern=r"^\d{6}$"
    )


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "CareerAI authentication",
        "email_configured": bool(
            SMTP_USERNAME and SMTP_PASSWORD
        )
    }


# ---------------------------------------------------------
# REQUEST OTP
# ---------------------------------------------------------

@app.post("/auth/request-otp")
def request_otp(req: OTPRequest):

    email = clean_email(str(req.email))
    name = req.name.strip()
    now = time.time()

    # Check SMTP configuration before creating an OTP
    if not SMTP_USERNAME or not SMTP_PASSWORD:

        raise HTTPException(
            status_code=500,
            detail=(
                "CareerAI email service is not configured. "
                "Please configure SMTP settings in .env."
            )
        )

    with db() as conn:

        # Prevent OTP spam
        recent = conn.execute(
            """
            SELECT created_at
            FROM otp_requests
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (email,)
        ).fetchone()

        if recent:

            elapsed = now - recent["created_at"]

            if elapsed < OTP_RESEND_SECONDS:

                wait = max(
                    1,
                    int(OTP_RESEND_SECONDS - elapsed)
                )

                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Please wait {wait} seconds "
                        "before requesting another OTP."
                    )
                )

        # Generate REAL random OTP
        otp = f"{secrets.randbelow(1_000_000):06d}"

        # Invalidate previous OTPs
        conn.execute(
            """
            UPDATE otp_requests
            SET used=1
            WHERE user_id=? AND used=0
            """,
            (email,)
        )

        # Store only hash of OTP
        conn.execute(
            """
            INSERT INTO otp_requests
            (
                user_id,
                otp_hash,
                created_at,
                expires_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                email,
                hash_value(otp),
                now,
                now + OTP_EXPIRY_SECONDS
            )
        )

        # Create/update user
        existing = conn.execute(
            """
            SELECT id
            FROM users
            WHERE user_id=?
            """,
            (email,)
        ).fetchone()

        if existing:

            conn.execute(
                """
                UPDATE users
                SET name=?
                WHERE user_id=?
                """,
                (name, email)
            )

        else:

            conn.execute(
                """
                INSERT INTO users
                (
                    name,
                    user_id,
                    created_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    email,
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )
            )

    # -----------------------------------------------------
    # SEND REAL EMAIL
    # -----------------------------------------------------

    try:

        send_email_otp(
            email=email,
            otp=otp,
            name=name
        )

    except Exception as e:

        # If email sending fails, invalidate OTP
        with db() as conn:

            conn.execute(
                """
                UPDATE otp_requests
                SET used=1
                WHERE user_id=? AND used=0
                """,
                (email,)
            )

        print("EMAIL ERROR:", repr(e))

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to send OTP email. "
                "Please check the email service configuration."
            )
        )

    # IMPORTANT:
    # NEVER return the OTP to the frontend.
    return {
        "message": "OTP sent successfully.",
        "expires_in_seconds": OTP_EXPIRY_SECONDS
    }


# ---------------------------------------------------------
# VERIFY OTP
# ---------------------------------------------------------

@app.post("/auth/verify-otp")
def verify_otp(req: OTPVerify):

    email = clean_email(str(req.email))
    name = req.name.strip()
    now = time.time()

    with db() as conn:

        row = conn.execute(
            """
            SELECT *
            FROM otp_requests
            WHERE user_id=?
              AND used=0
            ORDER BY id DESC
            LIMIT 1
            """,
            (email,)
        ).fetchone()

        if not row:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No active OTP found. "
                    "Please request a new OTP."
                )
            )

        # Check expiry
        if now > row["expires_at"]:

            conn.execute(
                """
                UPDATE otp_requests
                SET used=1
                WHERE id=?
                """,
                (row["id"],)
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "OTP has expired. "
                    "Please request a new OTP."
                )
            )

        # Check attempts
        if row["attempts"] >= MAX_OTP_ATTEMPTS:

            conn.execute(
                """
                UPDATE otp_requests
                SET used=1
                WHERE id=?
                """,
                (row["id"],)
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    "Too many incorrect attempts. "
                    "Please request a new OTP."
                )
            )

        # Compare submitted OTP with stored hash
        if not secrets.compare_digest(
            row["otp_hash"],
            hash_value(req.otp)
        ):

            conn.execute(
                """
                UPDATE otp_requests
                SET attempts=attempts+1
                WHERE id=?
                """,
                (row["id"],)
            )

            raise HTTPException(
                status_code=400,
                detail="Incorrect OTP. Please try again."
            )

        # OTP is valid
        conn.execute(
            """
            UPDATE otp_requests
            SET used=1
            WHERE id=?
            """,
            (row["id"],)
        )

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE user_id=?
            """,
            (email,)
        ).fetchone()

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User account was not found."
            )

        # Update login
        conn.execute(
            """
            UPDATE users
            SET name=?,
                last_login=?
            WHERE id=?
            """,
            (
                name,
                datetime.now(
                    timezone.utc
                ).isoformat(),
                user["id"]
            )
        )

        # Create session token
        token = secrets.token_urlsafe(32)

        conn.execute(
            """
            INSERT INTO sessions
            (
                token_hash,
                user_id,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                hash_value(token),
                user["id"],
                now
            )
        )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": name,
            "email": email
        }
    }


# ---------------------------------------------------------
# CURRENT USER
# ---------------------------------------------------------

@app.get("/auth/me")
def me(
    authorization: str | None = Header(default=None)
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Authentication required."
        )

    if not authorization.startswith("Bearer "):

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication header."
        )

    token = authorization[7:].strip()

    if not token:

        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token."
        )

    token_hash = hash_value(token)

    with db() as conn:

        row = conn.execute(
            """
            SELECT
                u.*
            FROM sessions s
            JOIN users u
              ON u.id=s.user_id
            WHERE s.token_hash=?
            """,
            (token_hash,)
        ).fetchone()

    if not row:

        raise HTTPException(
            status_code=401,
            detail="Invalid session."
        )

    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["user_id"]
    }