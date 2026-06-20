import sqlite3
import os
import logging
from datetime import datetime
import config

logger = logging.getLogger(__name__)

# Local threat intel directories
THREAT_INTEL_DIR = os.path.join(os.path.dirname(config.SECURITY_DB_PATH), "threat_intel")
REPUTATION_DB_PATH = os.path.join(THREAT_INTEL_DIR, "reputation.db")
INTERNETDB_PATH = os.path.join(THREAT_INTEL_DIR, "internetdb", "internetdb.db")
CVE_DB_PATH = os.path.join(THREAT_INTEL_DIR, "cvedb", "cve.db")

def initialize_intel():
    """Initializes local threat intelligence directories and SQLite databases with mock values for testing."""
    os.makedirs(THREAT_INTEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INTERNETDB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CVE_DB_PATH), exist_ok=True)

    # 1. Reputation Cache DB
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

    # 2. InternetDB mock database
    conn = sqlite3.connect(INTERNETDB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data (
            ip TEXT PRIMARY KEY,
            ports TEXT, -- comma-separated list, e.g. "80,443"
            hostnames TEXT, -- comma-separated list
            tags TEXT, -- comma-separated list
            vulns TEXT, -- comma-separated list of CVEs
            cpes TEXT -- comma-separated list
        )
    """)
    # Insert some mock values for testing
    cursor.executemany("""
        INSERT OR IGNORE INTO data (ip, ports, hostnames, tags, vulns, cpes)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        ("8.8.8.8", "53", "dns.google", "dns,safe", "", "cpe:/a:google:dns"),
        ("127.0.0.1", "22,80,443", "localhost", "local", "", ""),
        ("203.0.113.42", "22,8080", "compromised.host", "malicious,scanner", "CVE-2026-1234", "cpe:/o:linux:kernel")
    ])
    conn.commit()
    conn.close()

    # 3. CVE DB mock database
    conn = sqlite3.connect(CVE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vulns (
            cve_id TEXT PRIMARY KEY,
            cvss REAL,
            cvss_version TEXT,
            compressed_cve_data TEXT
        )
    """)
    # Insert some mock values
    cursor.executemany("""
        INSERT OR IGNORE INTO vulns (cve_id, cvss, cvss_version, compressed_cve_data)
        VALUES (?, ?, ?, ?)
    """, [
        ("CVE-2026-1234", 9.8, "3.1", "Critical remote code execution vulnerability in kernel process parser."),
        ("CVE-2026-5555", 7.5, "3.1", "High severity denial of service via malformed packet stream.")
    ])
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
    conn = sqlite3.connect(REPUTATION_DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO reputation_cache (ip, reputation_score, tags, last_updated)
            VALUES (?, ?, ?, ?)
        """, (ip, score, ",".join(tags), datetime.utcnow().isoformat() + "Z"))
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
    initialize_intel()
    try:
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
        logger.debug(f"GeoIP MMDB lookup failed or not configured: {e}")

    # Fallback default
    return {
        "country": "Unknown",
        "city": "Unknown",
        "latitude": None,
        "longitude": None
    }

def lookup_cve_details(cve_id: str) -> dict:
    """Looks up details of a CVE in the local SQLite CVEDB database."""
    initialize_intel()
    try:
        conn = sqlite3.connect(CVE_DB_PATH)
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
        if 'conn' in locals():
            conn.close()

    return {
        "cve_id": cve_id,
        "cvss": "N/A",
        "details": "Details not available locally."
    }

def lookup_internetdb(ip: str) -> dict:
    """Queries Shodan InternetDB SQLite database for open ports, hostnames, tags, vulns, and cpes."""
    initialize_intel()
    try:
        conn = sqlite3.connect(INTERNETDB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ports, hostnames, tags, vulns, cpes FROM data WHERE ip = ?", (ip,))
        row = cursor.fetchone()
        if row:
            return {
                "ip": ip,
                "ports": [int(p) for p in row[0].split(",") if p.strip().isdigit()] if row[0] else [],
                "hostnames": row[1].split(",") if row[1] else [],
                "tags": row[2].split(",") if row[2] else [],
                "vulns": row[3].split(",") if row[3] else [],
                "cpes": row[4].split(",") if row[4] else [],
                "found": True
            }
    except Exception as e:
        logger.error(f"Failed to query Shodan InternetDB: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    return {
        "ip": ip,
        "ports": [],
        "hostnames": [],
        "tags": [],
        "vulns": [],
        "cpes": [],
        "found": False
    }
