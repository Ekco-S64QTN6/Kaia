import hashlib
import sqlite3
from datetime import UTC, datetime
import uuid
import logging
import config

logger = logging.getLogger(__name__)

def initialize_db():
    """Initializes the append-only security events database."""
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload_hash TEXT,
            disposition TEXT NOT NULL,
            session_id TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consumed_capability_tokens (
            token_fingerprint TEXT PRIMARY KEY,
            expires_at_epoch REAL NOT NULL,
            consumed_at_epoch REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Security events database initialized.")


def consume_capability_token(signature: str, expires_at: datetime) -> tuple[bool, str | None]:
    """Atomically mark a signed token as used until its expiry.

    Only a one-way fingerprint of the HMAC signature is persisted. BEGIN IMMEDIATE
    serializes competing Policy Gate workers and the unique primary key makes a
    duplicate consume fail, including after a service restart. Fingerprints are not
    expired by wall-clock time: clock rollback must not reopen a used credential.
    A database error denies the protected operation.
    """
    if (
        not isinstance(signature, str)
        or len(signature) != 64
        or any(char not in "0123456789abcdef" for char in signature)
    ):
        return False, "Token signature is malformed."

    if (
        not isinstance(expires_at, datetime)
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
    ):
        return False, "Token expiry must be timezone-aware."

    try:
        expiry_epoch = expires_at.astimezone(UTC).timestamp()
        now_epoch = datetime.now(UTC).timestamp()
    except (OverflowError, OSError, ValueError) as exc:
        logger.error("Could not normalize capability-token expiry: %s", exc)
        return False, "Unable to enforce capability-token replay protection."

    if expiry_epoch <= now_epoch:
        return False, "Token has expired."

    fingerprint = hashlib.sha256(signature.encode("ascii")).hexdigest()
    conn = None
    try:
        conn = sqlite3.connect(config.SECURITY_DB_PATH, timeout=2.0)
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consumed_capability_tokens (
                token_fingerprint TEXT PRIMARY KEY,
                expires_at_epoch REAL NOT NULL,
                consumed_at_epoch REAL NOT NULL
            )
        """)
        # Never prune fingerprints based on token expiry or wall-clock time.
        # Clock rollback must not make an already-used token usable again.
        cursor.execute(
            """
            INSERT OR IGNORE INTO consumed_capability_tokens
                (token_fingerprint, expires_at_epoch, consumed_at_epoch)
            VALUES (?, ?, ?)
            """,
            (fingerprint, expiry_epoch, now_epoch),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return False, "Token has already been used."

        conn.commit()
        return True, None
    except sqlite3.Error as exc:
        if conn is not None:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
        logger.error("Capability-token replay protection failed closed: %s", exc)
        return False, "Unable to enforce capability-token replay protection; request denied."
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error as exc:
                logger.error("Failed to close capability-token replay database: %s", exc)


def log_security_event(event_type: str, source: str, actor: str, payload_hash: str, disposition: str, session_id: str) -> str:
    """
    Appends a new security event.
    Returns the generated event_id.
    """
    from datetime import UTC
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO security_events (event_id, timestamp, type, source, actor, payload_hash, disposition, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, timestamp, event_type, source, actor, payload_hash, disposition, session_id))
        conn.commit()
        logger.warning(f"Security event recorded: {event_id} | Type: {event_type} | Disposition: {disposition}")
    except Exception as e:
        logger.error(f"Failed to log security event: {e}")
    finally:
        conn.close()
        
    return event_id

def query_security_events(limit: int = 100) -> list:
    """
    Queries security events. Accessible for analysis but not modifiable by the agent.
    """
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT event_id, timestamp, type, source, actor, payload_hash, disposition, session_id
            FROM security_events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        return results
    except Exception as e:
        logger.error(f"Failed to query security events: {e}")
        return []
    finally:
        conn.close()
