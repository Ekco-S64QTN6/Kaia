#!/usr/bin/env python3
"""Download the MaxMind GeoLite2 archive with a hard streamed byte limit."""
import os
import re
import sys
import requests

GEOIP_DOWNLOAD_URL = "https://download.maxmind.com/geoip/databases/GeoLite2-City/download?suffix=tar.gz"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024

def download_archive(output_path, account_id=None, license_key=None):
    """Download one non-empty archive to an exclusively created private file."""
    account_id = os.environ.get("MAXMIND_ACCOUNT_ID", "") if account_id is None else account_id
    license_key = os.environ.get("MAXMIND_LICENSE_KEY", "") if license_key is None else license_key
    if not isinstance(account_id, str) or not re.fullmatch(r"[0-9]+", account_id):
        raise ValueError("MaxMind account ID must contain only digits")
    if not isinstance(license_key, str) or not re.fullmatch(r"[A-Za-z0-9]+", license_key):
        raise ValueError("MaxMind license key must contain only alphanumeric characters")
    response = None
    created = False
    output_path = os.fspath(output_path)
    try:
        response = requests.get(GEOIP_DOWNLOAD_URL, auth=(account_id, license_key), timeout=(10, 60), stream=True)
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except (TypeError, ValueError) as exc:
                raise ValueError("GeoIP response has an invalid Content-Length") from exc
            if declared_length < 0 or declared_length > MAX_ARCHIVE_BYTES:
                raise ValueError(f"GeoIP archive exceeds the {MAX_ARCHIVE_BYTES}-byte download limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(output_path, flags, 0o600)
        created = True
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as output:
                fd = -1
                total = 0
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                    if not chunk:
                        continue
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise ValueError("GeoIP response yielded a non-byte chunk")
                    if total + len(chunk) > MAX_ARCHIVE_BYTES:
                        raise ValueError(f"GeoIP archive exceeds the {MAX_ARCHIVE_BYTES}-byte download limit")
                    output.write(chunk)
                    total += len(chunk)
                if total == 0:
                    raise ValueError("GeoIP download is empty")
                output.flush()
                os.fsync(output.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
        return total
    except Exception:
        if created:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
        raise
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                close()

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("Usage: download_geoip_archive.py OUTPUT_PATH", file=sys.stderr)
        return 2
    try:
        total = download_archive(argv[0])
    except Exception as exc:
        print(f"Error: GeoIP download rejected: {exc}", file=sys.stderr)
        return 1
    print(f"Downloaded {total} bytes of GeoIP archive.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
