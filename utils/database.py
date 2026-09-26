# utils/database.py — SQLite database layer with password auth

import sqlite3, os, json, time
from werkzeug.security import generate_password_hash, check_password_hash
from config import DB_PATH, DATA_DIR


def get_conn():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            pw_hash     TEXT NOT NULL,
            mfa_secret  TEXT,
            enrolled    INTEGER DEFAULT 0,
            created_at  REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS enrolment_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            features    TEXT NOT NULL,
            passage_num INTEGER DEFAULT 0,
            collected_at REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS session_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            session_id  TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            trust_score REAL,
            details     TEXT,
            timestamp   REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS eval_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            label       TEXT NOT NULL,
            features    TEXT NOT NULL,
            created_at  REAL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # ── Admin column migration (safe on existing DBs) ──────────────────────
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()
    conn.close()
    print("[DB] Initialised.")


# ── User CRUD ──────────────────────────────────────────────────────────────────

def create_user(username, password, mfa_secret):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, pw_hash, mfa_secret, enrolled, created_at) VALUES (?,?,?,0,?)",
            (username, generate_password_hash(password), mfa_secret, time.time())
        )
        conn.commit()
        return conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()["id"]
    finally:
        conn.close()


def verify_password(username, password):
    conn = get_conn()
    row = conn.execute("SELECT pw_hash FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row is None:
        return False
    return check_password_hash(row["pw_hash"], password)


def get_user(username):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(uid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_enrolled(user_id, val=True):
    conn = get_conn()
    conn.execute("UPDATE users SET enrolled=? WHERE id=?", (1 if val else 0, user_id))
    conn.commit()
    conn.close()


# ── Enrolment samples ─────────────────────────────────────────────────────────

def save_enrolment_window(user_id, features, passage_num=0):
    conn = get_conn()
    conn.execute(
        "INSERT INTO enrolment_samples (user_id, features, passage_num, collected_at) VALUES (?,?,?,?)",
        (user_id, json.dumps(features), passage_num, time.time())
    )
    conn.commit()
    conn.close()


def get_enrolment_samples(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT features FROM enrolment_samples WHERE user_id=? ORDER BY collected_at",
        (user_id,)
    ).fetchall()
    conn.close()
    return [json.loads(r["features"]) for r in rows]


def count_enrolment_windows(user_id):
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) as c FROM enrolment_samples WHERE user_id=?", (user_id,)).fetchone()["c"]
    conn.close()
    return n


def count_free_enrolment(user_id):
    """Free-text enrolment windows are stored with passage_num = -1."""
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) as c FROM enrolment_samples WHERE user_id=? AND passage_num=-1",
        (user_id,)).fetchone()["c"]
    conn.close()
    return n


def clear_enrolment(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM enrolment_samples WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


# ── Evaluation samples (live FAR/FRR protocol) ────────────────────────────────

def save_eval_samples(user_id, label, feature_vectors):
    """Store a batch of window feature vectors under a label, e.g.
    'genuine', 'impostor_fast', 'impostor_slow', 'bot'."""
    conn = get_conn()
    conn.executemany(
        "INSERT INTO eval_samples (user_id, label, features, created_at) VALUES (?,?,?,?)",
        [(user_id, label, json.dumps(fv), time.time()) for fv in feature_vectors]
    )
    conn.commit()
    conn.close()


def get_eval_samples(user_id):
    """Returns list of dicts {label, features} for a user."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT label, features FROM eval_samples WHERE user_id=? ORDER BY created_at",
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"label": r["label"], "features": json.loads(r["features"])} for r in rows]


def count_eval_samples(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT label, COUNT(*) AS c FROM eval_samples WHERE user_id=? GROUP BY label",
        (user_id,)
    ).fetchall()
    conn.close()
    return {r["label"]: r["c"] for r in rows}


def clear_eval_samples(user_id):
    conn = get_conn()
    conn.execute("DELETE FROM eval_samples WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


# ── Session events ─────────────────────────────────────────────────────────────

def log_event(user_id, session_id, event_type, trust_score=None, details=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO session_events (user_id, session_id, event_type, trust_score, details, timestamp) VALUES (?,?,?,?,?,?)",
        (user_id, session_id, event_type, trust_score, details, time.time())
    )
    conn.commit()
    conn.close()


def get_recent_events(user_id, limit=25):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM session_events WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Admin functions ────────────────────────────────────────────────────────────

def get_all_users():
    """Return all users with enrolment status and event/anomaly counts."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT u.*,
               COALESCE(e.total_events, 0)  AS total_events,
               COALESCE(e.anomaly_count, 0) AS anomaly_count,
               e.last_activity
        FROM users u
        LEFT JOIN (
            SELECT user_id,
                   COUNT(*)                                     AS total_events,
                   SUM(CASE WHEN event_type IN ('warning','mfa_trigger','session_lock') THEN 1 ELSE 0 END) AS anomaly_count,
                   MAX(timestamp)                               AS last_activity
            FROM session_events
            GROUP BY user_id
        ) e ON u.id = e.user_id
        ORDER BY u.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_admin(user_id, val=True):
    conn = get_conn()
    conn.execute("UPDATE users SET is_admin=? WHERE id=?", (1 if val else 0, user_id))
    conn.commit()
    conn.close()


def get_user_anomaly_summary(user_id):
    """Aggregate abnormal events for a specific user."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT event_type, COUNT(*) AS cnt,
               AVG(trust_score) AS avg_score,
               MIN(trust_score) AS min_score,
               MAX(timestamp)   AS last_occurrence
        FROM session_events
        WHERE user_id=? AND event_type IN ('warning','mfa_trigger','mfa_fail','session_lock')
        GROUP BY event_type
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_anomaly_events(limit=100):
    """System-wide abnormal events feed."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT se.*, u.username
        FROM session_events se
        JOIN users u ON se.user_id = u.id
        WHERE se.event_type IN ('warning','mfa_trigger','mfa_fail','session_lock','mfa_pass')
        ORDER BY se.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_trust_history(user_id, limit=200):
    """Trust score history for charting."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT trust_score, timestamp, event_type
        FROM session_events
        WHERE user_id=? AND trust_score IS NOT NULL
        ORDER BY timestamp ASC
        LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_user_all_events(user_id, limit=200):
    """All session events for a user, most recent first."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM session_events WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user_full(user_id):
    """Delete a user and all associated data."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM enrolment_samples WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM eval_samples WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM session_events WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"[DB] Error in delete_user_full: {e}")
        raise e
    finally:
        conn.close()



def get_system_stats():
    """Dashboard statistics."""
    conn = get_conn()
    c = conn.cursor()
    total_users = c.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    enrolled_users = c.execute("SELECT COUNT(*) AS c FROM users WHERE enrolled=1").fetchone()["c"]
    total_events = c.execute("SELECT COUNT(*) AS c FROM session_events").fetchone()["c"]

    import time as _time
    day_ago = _time.time() - 86400
    anomalies_24h = c.execute(
        "SELECT COUNT(*) AS c FROM session_events WHERE event_type IN ('warning','mfa_trigger','session_lock') AND timestamp>?",
        (day_ago,)
    ).fetchone()["c"]
    mfa_triggers_24h = c.execute(
        "SELECT COUNT(*) AS c FROM session_events WHERE event_type='mfa_trigger' AND timestamp>?",
        (day_ago,)
    ).fetchone()["c"]
    conn.close()
    return {
        "total_users": total_users,
        "enrolled_users": enrolled_users,
        "total_events": total_events,
        "anomalies_24h": anomalies_24h,
        "mfa_triggers_24h": mfa_triggers_24h,
    }

