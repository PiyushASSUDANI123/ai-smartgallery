import os
import psycopg2
from psycopg2.extras import DictCursor
import hashlib
import numpy as np
import logging
from contextlib import contextmanager
from app.core.config import settings

logger = logging.getLogger("Database")

DB_URL = settings.DATABASE_URL

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class PostgresCursorWrapper:
    def __init__(self, cursor):
        self.cursor = cursor
        
    def fetchone(self):
        row = self.cursor.fetchone()
        return dict(row) if row else None
        
    def fetchall(self):
        rows = self.cursor.fetchall()
        return [dict(r) for r in rows] if rows else []
        
    @property
    def rowcount(self):
        return self.cursor.rowcount

class PostgresConnectionWrapper:
    def __init__(self, conn):
        self.conn = conn
        
    def execute(self, query: str, parameters=None):
        pg_query = query.replace("?", "%s")
        cursor = self.conn.cursor(cursor_factory=DictCursor)
        if parameters:
            cursor.execute(pg_query, parameters)
        else:
            cursor.execute(pg_query)
        return PostgresCursorWrapper(cursor)
        
    def commit(self):
        self.conn.commit()
        
    def rollback(self):
        self.conn.rollback()
        
    def close(self):
        self.conn.close()
        
    def cursor(self):
        # Return a raw psycopg2 cursor for init_db
        return self.conn.cursor()

@contextmanager
def get_db():
    conn = None
    try:
        conn = psycopg2.connect(DB_URL)
        yield PostgresConnectionWrapper(conn)
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database transaction failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    logger.info(f"Initializing Supabase PostgreSQL database.")
    with get_db() as wrapper:
        conn = wrapper.conn
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK(role IN ('owner', 'junior', 'superadmin')) NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    plan_name TEXT DEFAULT 'Basic',
                    allocated_storage_bytes BIGINT DEFAULT 53687091200,
                    subscription_expires_at TEXT,
                    processing_priority TEXT DEFAULT 'normal',
                    custom_storage_bytes BIGINT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    client_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    auto_expiry INTEGER DEFAULT 0,
                    pin TEXT,
                    logo_filename TEXT,
                    watermark_enabled INTEGER DEFAULT 0,
                    paywall_enabled INTEGER DEFAULT 0,
                    owner_username TEXT REFERENCES users(username) ON DELETE CASCADE,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_photos (
                    id SERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    file_url TEXT NOT NULL,
                    file_size BIGINT DEFAULT 0,
                    faces_scanned INTEGER DEFAULT 0,
                    faces_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id SERIAL PRIMARY KEY,
                    photo_id INTEGER NOT NULL REFERENCES event_photos(id) ON DELETE CASCADE,
                    encoding BYTEA NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id SERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    metric_type TEXT NOT NULL,
                    visitor_ip TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id SERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id SERIAL PRIMARY KEY,
                    lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
                    photo_id INTEGER NOT NULL REFERENCES event_photos(id) ON DELETE CASCADE,
                    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                    matched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lead_id, photo_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS photo_selections (
                    photo_id INTEGER PRIMARY KEY REFERENCES event_photos(id) ON DELETE CASCADE,
                    selected INTEGER DEFAULT 1,
                    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    storage_limit_gb INTEGER NOT NULL,
                    event_limit INTEGER NOT NULL,
                    price_inr INTEGER NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT,
                    action TEXT NOT NULL,
                    duration_ms INTEGER,
                    payload_size INTEGER,
                    status TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS broadcasts (
                    id SERIAL PRIMARY KEY,
                    message TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # --- PERFORMANCE OPTIMIZATION: B-Tree Indexes ---
            logger.info("Creating B-Tree Indexes for rapid lookups...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_owner ON events(owner_username);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_event ON event_photos(event_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON face_encodings(photo_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analytics_event ON analytics(event_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_lead ON matches(lead_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_event ON matches(event_id);")

            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            if user_count == 0:
                super_user = "piyush"
                super_pass = hash_password("1234")
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role, name) VALUES (%s, %s, 'superadmin', 'Super Administrator')",
                    (super_user, super_pass)
                )
                default_user = "admin"
                default_pass = hash_password("admin123")
                cursor.execute(
                    "INSERT INTO users (username, password_hash, role, name) VALUES (%s, %s, 'owner', 'Administrator')",
                    (default_user, default_pass)
                )
                logger.info("Seeded default superadmin and owner")
            else:
                cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'piyush'")
                if cursor.fetchone()[0] == 0:
                    super_pass = hash_password("1234")
                    cursor.execute(
                        "INSERT INTO users (username, password_hash, role, name) VALUES ('piyush', %s, 'superadmin', 'Super Administrator')",
                        (super_pass,)
                    )

            cursor.execute("SELECT COUNT(*) FROM plans")
            if cursor.fetchone()[0] == 0:
                cursor.execute("INSERT INTO plans (name, storage_limit_gb, event_limit, price_inr) VALUES ('Basic', 50, 10, 999)")
                cursor.execute("INSERT INTO plans (name, storage_limit_gb, event_limit, price_inr) VALUES ('Pro', 200, 9999, 3499)")

        logger.info("Database schemas initialized successfully in Supabase PostgreSQL.")

def serialize_encoding(encoding: np.ndarray) -> bytes:
    return encoding.astype(np.float64).tobytes()

def deserialize_encoding(encoding_bytes: bytes) -> np.ndarray:
    return np.frombuffer(encoding_bytes, dtype=np.float64)
