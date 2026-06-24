import sqlite3
import os
import logging
import time
from datetime import datetime, timedelta

import requests

import config

logger = logging.getLogger(__name__)

# Local threat intel directories
THREAT_INTEL_DIR = os.path.join(os.path.dirname(config.SECURITY_DB_PATH), "threat_intel")
REPUTATION_DB_PATH = os.path.join(THREAT_INTEL_DIR, "reputation.db")
INTERNETDB_PATH = os.path.join(THREAT_INTEL_DIR, "internetdb", "internetdb.db")
CVE_DB_PATH = os.path.join(THREAT_INTEL_DIR, "cvedb", "cve.db")

# InternetDB API configuration
INTERNETDB_API_URL = "https://internetdb.shodan.io"
INTERNETDB_CACHE_TTL_DAYS = 7
_last_api_call_time = 0.0  # Module-level rate limiter state

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

    # 2. InternetDB cache database
    conn = sqlite3.connect(INTERNETDB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS data (
            ip TEXT PRIMARY KEY,
            ports TEXT, -- comma-separated list, e.g. "80,443"
            hostnames TEXT, -- comma-separated list
            tags TEXT, -- comma-separated list
            vulns TEXT, -- comma-separated list of CVEs
            cpes TEXT, -- comma-separated list
            last_updated TEXT NOT NULL DEFAULT ''
        )
    """)
    # Migration: add last_updated column if missing from an older schema
    try:
        cursor.execute("SELECT last_updated FROM data LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE data ADD COLUMN last_updated TEXT NOT NULL DEFAULT ''")
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

def _is_cache_stale(last_updated_str: str) -> bool:
    """Returns True if the cached entry is older than INTERNETDB_CACHE_TTL_DAYS or has no timestamp."""
    if not last_updated_str:
        return True
    try:
        cached_at = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00").replace("+00:00", ""))
        return (datetime.utcnow() - cached_at) > timedelta(days=INTERNETDB_CACHE_TTL_DAYS)
    except (ValueError, TypeError):
        return True


def _fetch_internetdb_api(ip: str) -> dict:
    """
    Fetches IP enrichment data from the free Shodan InternetDB API.
    https://internetdb.shodan.io/{ip} — no API key required, free for non-commercial use.
    Rate-limited to ~1 request/second.
    Returns parsed dict on success, empty dict on failure.
    """
    global _last_api_call_time

    # Enforce ~1 req/sec rate limit
    elapsed = time.monotonic() - _last_api_call_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    url = f"{INTERNETDB_API_URL}/{ip}"
    try:
        resp = requests.get(url, timeout=10)
        _last_api_call_time = time.monotonic()

        if resp.status_code == 404:
            # Shodan returns 404 for IPs with no data — this is expected, not an error
            logger.debug(f"InternetDB: No data for IP {ip} (404)")
            return {}
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        _last_api_call_time = time.monotonic()
        logger.warning(f"InternetDB API request failed for {ip}: {e}")
        return {}


def _cache_internetdb_result(ip: str, api_data: dict):
    """Writes an InternetDB API response into the local SQLite cache."""
    try:
        conn = sqlite3.connect(INTERNETDB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO data (ip, ports, hostnames, tags, vulns, cpes, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ip,
            ",".join(str(p) for p in api_data.get("ports", [])),
            ",".join(api_data.get("hostnames", [])),
            ",".join(api_data.get("tags", [])),
            ",".join(api_data.get("vulns", [])),
            ",".join(api_data.get("cpes", [])),
            datetime.utcnow().isoformat() + "Z"
        ))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to cache InternetDB result for {ip}: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


def _parse_internetdb_row(ip: str, row: tuple, source: str = "cache") -> dict:
    """Parses a raw SQLite row into the standard InternetDB result dict."""
    return {
        "ip": ip,
        "ports": [int(p) for p in row[0].split(",") if p.strip().isdigit()] if row[0] else [],
        "hostnames": [h for h in row[1].split(",") if h] if row[1] else [],
        "tags": [t for t in row[2].split(",") if t] if row[2] else [],
        "vulns": [v for v in row[3].split(",") if v] if row[3] else [],
        "cpes": [c for c in row[4].split(",") if c] if row[4] else [],
        "found": True,
        "source": source
    }


def lookup_internetdb(ip: str) -> dict:
    """
    Queries Shodan InternetDB for open ports, hostnames, tags, vulns, and CPEs.

    Strategy: cache-first with lazy on-demand API fallback.
    1. Check local SQLite cache.
    2. On cache miss or stale entry (>7 days), call the free InternetDB API.
    3. Cache the API result locally for future lookups.
    4. On API error, return the stale cache if available, otherwise empty result.
    """
    initialize_intel()
    _NOT_FOUND = {
        "ip": ip, "ports": [], "hostnames": [], "tags": [],
        "vulns": [], "cpes": [], "found": False, "source": "none"
    }

    # Step 1: Check local cache
    cached_row = None
    cached_timestamp = ""
    try:
        conn = sqlite3.connect(INTERNETDB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT ports, hostnames, tags, vulns, cpes, last_updated FROM data WHERE ip = ?", (ip,))
        row = cursor.fetchone()
        if row:
            cached_row = row[:5]
            cached_timestamp = row[5] if row[5] else ""
    except Exception as e:
        logger.error(f"Failed to query InternetDB cache: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

    # Step 2: Return fresh cache hit immediately
    if cached_row and not _is_cache_stale(cached_timestamp):
        return _parse_internetdb_row(ip, cached_row, source="cache")

    # Step 3: Cache miss or stale — call the API
    api_data = _fetch_internetdb_api(ip)

    if api_data:
        # Cache the fresh result
        _cache_internetdb_result(ip, api_data)
        return {
            "ip": ip,
            "ports": api_data.get("ports", []),
            "hostnames": api_data.get("hostnames", []),
            "tags": api_data.get("tags", []),
            "vulns": api_data.get("vulns", []),
            "cpes": api_data.get("cpes", []),
            "found": True,
            "source": "api"
        }

    # Step 4: API failed — return stale cache if available
    if cached_row:
        logger.info(f"InternetDB API unavailable for {ip}, returning stale cache")
        return _parse_internetdb_row(ip, cached_row, source="stale_cache")

    return _NOT_FOUND
