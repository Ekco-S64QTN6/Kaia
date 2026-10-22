import sqlite3
import ipaddress
import threading
import os
import logging
import time
from datetime import datetime, timedelta, UTC

import requests

import config

logger = logging.getLogger(__name__)

# Local threat intel directories
THREAT_INTEL_DIR = str(config.THREAT_INTEL_DIR)
REPUTATION_DB_PATH = os.path.join(THREAT_INTEL_DIR, "reputation.db")
INTERNETDB_PATH = os.path.join(THREAT_INTEL_DIR, "internetdb", "internetdb.db")
CVE_DB_PATH = os.path.join(THREAT_INTEL_DIR, "cvedb", "cve.db")

# InternetDB API configuration
INTERNETDB_API_URL = "https://internetdb.shodan.io"
INTERNETDB_CACHE_TTL_DAYS = 7
_last_api_call_time = 0.0  # Monotonic start time for the most recent request
_api_rate_limit_lock = threading.Lock()

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
            last_updated TEXT NOT NULL DEFAULT '',
            found INTEGER NOT NULL DEFAULT 1
        )
    """)
    # Migrations for caches created by older versions.
    try:
        cursor.execute("SELECT last_updated FROM data LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE data ADD COLUMN last_updated TEXT NOT NULL DEFAULT ''")
    try:
        cursor.execute("SELECT found FROM data LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE data ADD COLUMN found INTEGER NOT NULL DEFAULT 1")
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
        """, (ip, score, ",".join(tags), datetime.now(UTC).isoformat().replace("+00:00", "Z")))
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
        import maxminddb
        geoip_db = config.GEOIP_DB_PATH
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
        expires_str = last_updated_str
        if expires_str.endswith("Z"):
            expires_str = expires_str[:-1] + "+00:00"
        cached_at = datetime.fromisoformat(expires_str)
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=UTC)
        return (datetime.now(UTC) - cached_at) > timedelta(days=INTERNETDB_CACHE_TTL_DAYS)
    except (ValueError, TypeError):
        return True


def _fetch_internetdb_api(ip: str) -> dict:
    """
    Fetch InternetDB while enforcing the per-process start-rate limit.

    Requests are serialized through completion. This is deliberately conservative:
    a slow response reduces throughput, but two threads cannot reserve a slot and
    then be scheduled together into requests.get().
    """
    global _last_api_call_time
    try:
        address = ipaddress.ip_address(str(ip))
        ip = address.compressed
    except (ValueError, TypeError):
        return {}

    # InternetDB is a public-host intelligence API, not an internal asset
    # resolver. Never disclose private, loopback, link-local, reserved or
    # otherwise non-global addresses to this third-party service.
    if not address.is_global:
        logger.debug("Skipping InternetDB request for non-global address %s", ip)
        return {}

    url = f"{INTERNETDB_API_URL}/{ip}"
    with _api_rate_limit_lock:
        elapsed = time.monotonic() - _last_api_call_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        _last_api_call_time = time.monotonic()

        try:
            # Keep request initiation within the serialized critical section.
            # requests.get is bounded by timeout and returns only after receiving
            # the response body with the default non-streaming behavior.
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                logger.debug("InternetDB: No data for %s (404)", ip)
                return {"_kaia_not_found": True}
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("InternetDB returned a non-object payload for %s", ip)
                return {}
            return data
        except (requests.RequestException, ValueError, TypeError) as e:
            logger.warning("InternetDB API request failed for %s: %s", ip, e)
            return {}

def _cache_internetdb_result(ip: str, api_data: dict, found: bool = True):
    """Cache a validated InternetDB result, including authoritative 404/no-data results."""
    conn = None
    try:
        conn = sqlite3.connect(INTERNETDB_PATH, timeout=2.0)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO data
                (ip, ports, hostnames, tags, vulns, cpes, last_updated, found)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ip,
                ",".join(str(p) for p in api_data.get("ports", []) if isinstance(p, (int, str))),
                ",".join(str(h) for h in api_data.get("hostnames", []) if isinstance(h, str)),
                ",".join(str(t) for t in api_data.get("tags", []) if isinstance(t, str)),
                ",".join(str(v) for v in api_data.get("vulns", []) if isinstance(v, str)),
                ",".join(str(c) for c in api_data.get("cpes", []) if isinstance(c, str)),
                datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                1 if found else 0,
            ),
        )
        conn.commit()
    except Exception as e:
        logger.error("Failed to cache InternetDB result for %s: %s", ip, e)
    finally:
        if conn:
            conn.close()

def _parse_internetdb_row(ip: str, row: tuple, source: str = "cache", found: bool = True) -> dict:
    """Parse cached fields while preserving whether the API reported a match."""
    return {
        "ip": ip,
        "ports": [int(p) for p in row[0].split(",") if p.strip().isdigit()] if row[0] else [],
        "hostnames": [h for h in row[1].split(",") if h] if row[1] else [],
        "tags": [t for t in row[2].split(",") if t] if row[2] else [],
        "vulns": [v for v in row[3].split(",") if v] if row[3] else [],
        "cpes": [c for c in row[4].split(",") if c] if row[4] else [],
        "found": bool(found),
        "source": source,
    }

def lookup_internetdb(ip: str) -> dict:
    """
    Cache-first InternetDB lookup with bounded input and negative caching.

    Authoritative 404 results are cached for the same TTL as positive results;
    transient request/parse failures are never cached as authoritative misses.
    """
    try:
        address = ipaddress.ip_address(str(ip))
        ip = address.compressed
    except (ValueError, TypeError):
        logger.warning("Refusing invalid InternetDB IP address: %r", ip)
        return {
            "ip": str(ip)[:64], "ports": [], "hostnames": [], "tags": [],
            "vulns": [], "cpes": [], "found": False, "source": "invalid",
        }

    if not address.is_global:
        logger.debug("Skipping InternetDB lookup for non-global address %s", ip)
        return {
            "ip": ip, "ports": [], "hostnames": [], "tags": [],
            "vulns": [], "cpes": [], "found": False, "source": "non-global",
        }

    initialize_intel()
    not_found = {
        "ip": ip, "ports": [], "hostnames": [], "tags": [],
        "vulns": [], "cpes": [], "found": False, "source": "none",
    }

    cached_row = None
    cached_timestamp = ""
    cached_found = True
    conn = None
    try:
        conn = sqlite3.connect(INTERNETDB_PATH, timeout=2.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ports, hostnames, tags, vulns, cpes, last_updated, found FROM data WHERE ip = ?",
            (ip,),
        )
        row = cursor.fetchone()
        if row:
            cached_row = row[:5]
            cached_timestamp = row[5] or ""
            cached_found = bool(row[6])
    except Exception as e:
        logger.error("Failed to query InternetDB cache: %s", e)
    finally:
        if conn:
            conn.close()

    if cached_row is not None and not _is_cache_stale(cached_timestamp):
        return _parse_internetdb_row(ip, cached_row, source="cache", found=cached_found)

    api_data = _fetch_internetdb_api(ip)

    # Only a real HTTP 404 is a negative cache entry. Network, TLS, status and
    # JSON parsing errors return {}, which must not erase or poison older data.
    if api_data.get("_kaia_not_found") is True:
        _cache_internetdb_result(ip, {}, found=False)
        return {**not_found, "source": "api-not-found"}

    if api_data:
        normalized = {}
        for field in ("ports", "hostnames", "tags", "vulns", "cpes"):
            value = api_data.get(field, [])
            if not isinstance(value, list):
                logger.warning("Ignoring malformed InternetDB field %s for %s", field, ip)
                value = []
            normalized[field] = value[:256]
        _cache_internetdb_result(ip, normalized, found=True)
        return {
            "ip": ip,
            **normalized,
            "found": True,
            "source": "api",
        }

    if cached_row is not None:
        logger.info("InternetDB API unavailable for %s, returning stale cache", ip)
        return _parse_internetdb_row(
            ip, cached_row, source="stale_cache", found=cached_found
        )

    return not_found
