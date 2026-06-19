import sqlite3
from datetime import datetime
import uuid
import logging
import config

logger = logging.getLogger(__name__)

def initialize_db():
    """Initializes the append-only security events database."""
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()
    logger.info("Security events database initialized.")

def log_security_event(event_type: str, source: str, actor: str, payload_hash: str, disposition: str, session_id: str) -> str:
    """
    Appends a new security event.
    Returns the generated event_id.
    """
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.utcnow().isoformat() + "Z"
    
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
