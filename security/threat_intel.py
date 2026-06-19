import sqlite3
import os
import logging
import config

logger = logging.getLogger(__name__)

# Local threat intel dir
THREAT_INTEL_DIR = os.path.join(os.path.dirname(config.SECURITY_DB_PATH), "threat_intel")
REPUTATION_DB_PATH = os.path.join(THREAT_INTEL_DIR, "reputation.db")

def initialize_intel():
    """Initializes local threat intelligence folders and reputation DB."""
    os.makedirs(THREAT_INTEL_DIR, exist_ok=True)
    conn = sqlite3.connect(REPUTATION_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reputation_cache (
            ip TEXT PRIMARY KEY,
            reputation_score INTEGER NOT NULL, -- 0 (malicious) to 100 (clean)
            tags TEXT,
            last_updated TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_ip_reputation(ip: str) -> dict:
    """Queries reputation database for IP."""
    initialize_intel()
    conn = sqlite3.connect(REPUTATION_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT reputation_score, tags, last_updated FROM reputation_cache WHERE ip = ?", (ip,))
        row = cursor.fetchone()
        if row:
            return {
                "ip": ip,
                "score": row[0],
                "tags": row[1].split(",") if row[1] else [],
                "cached": True
            }
    except Exception as e:
        logger.error(f"Failed to query local reputation cache: {e}")
    finally:
        conn.close()

    # Default fallback reputation if not cached
    return {
        "ip": ip,
        "score": 100,
        "tags": ["unrated"],
        "cached": False
    }

def update_ip_reputation(ip: str, score: int, tags: list):
    """Updates the local reputation database."""
    initialize_intel()
    from datetime import datetime
    conn = sqlite3.connect(REPUTATION_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO reputation_cache (ip, reputation_score, tags, last_updated)
            VALUES (?, ?, ?, ?)
        """, (ip, score, ",".join(tags), datetime.utcnow().isoformat()))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update reputation cache: {e}")
    finally:
        conn.close()

def lookup_geoip(ip: str) -> dict:
    """
    Attempts to read from MaxMind GeoLite2-City MMDB database.
    If database is missing, returns safe default fields.
    """
    try:
        # Check if database reader is installed
        import maxminddb
        geoip_db = os.path.join(THREAT_INTEL_DIR, "geoip", "GeoLite2-City.mmdb")
        if os.path.exists(geoip_db):
            with maxminddb.open_database(geoip_db) as reader:
                record = reader.get(ip)
                if record:
                    return {
                        "country": record.get("country", {}).get("names", {}).get("en", "Unknown"),
                        "city": record.get("city", {}).get("names", {}).get("en", "Unknown"),
                        "latitude": record.get("location", {}).get("latitude"),
                        "longitude": record.get("location", {}).get("longitude")
                    }
    except Exception as e:
        logger.debug(f"GeoIP MMDB lookup not fully initialized or error: {e}")

    # Fallback default
    return {
        "country": "Unknown",
        "city": "Unknown",
        "latitude": None,
        "longitude": None
    }

def lookup_cve_details(cve_id: str) -> dict:
    """Looks up details of a CVE in the local SQLite CVEDB database."""
    cve_db_path = os.path.join(THREAT_INTEL_DIR, "cvedb", "cve.db")
    if os.path.exists(cve_db_path):
        try:
            conn = sqlite3.connect(cve_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT cvss, compressed_cve_data FROM vulns WHERE cve_id = ?", (cve_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "cve_id": cve_id,
                    "cvss": row[0],
                    "details": row[1]
                }
        except Exception as e:
            logger.error(f"Failed to query local CVE db: {e}")
        finally:
            if 'conn' in locals(): conn.close()

    # Generic info fallback
    return {
        "cve_id": cve_id,
        "cvss": "N/A",
        "details": "Details not available locally."
    }
