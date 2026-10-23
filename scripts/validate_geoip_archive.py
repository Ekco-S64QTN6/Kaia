#!/usr/bin/env python3
"""Validate and extract one GeoLite2 City MMDB from a strictly bounded tar.gz."""
import os
import stat
import sys
import tarfile
from pathlib import PurePosixPath

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_EXPANDED_MEMBER_BYTES = 640 * 1024 * 1024
MAX_MMDB_BYTES = 512 * 1024 * 1024
EXTRACT_CHUNK_BYTES = 1024 * 1024

def validate_and_extract(archive_path, output_path):
    """Stream-validate every tar member and exclusively extract the unique MMDB."""
    output_path = os.fspath(output_path)
    output_created = False
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        archive_fd = os.open(os.fspath(archive_path), flags)
        with os.fdopen(archive_fd, "rb") as archive_file:
            archive_stat = os.fstat(archive_file.fileno())
            if not stat.S_ISREG(archive_stat.st_mode):
                raise ValueError("GeoIP archive is not a regular file")
            if archive_stat.st_size <= 0 or archive_stat.st_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f"GeoIP archive must be non-empty and at most {MAX_ARCHIVE_BYTES} bytes")
            member_count = 0
            expanded_bytes = 0
            candidate_count = 0
            extracted_bytes = 0
            with tarfile.open(fileobj=archive_file, mode="r|gz") as archive:
                for member in archive:
                    member_count += 1
                    if member_count > MAX_ARCHIVE_MEMBERS:
                        raise ValueError(f"GeoIP archive exceeds the {MAX_ARCHIVE_MEMBERS}-member limit")
                    if not isinstance(member.name, str) or "\\" in member.name or "\x00" in member.name:
                        raise ValueError("GeoIP archive contains an invalid member path")
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("GeoIP archive contains an absolute or parent-traversal path")
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("GeoIP archive contains a link or special file")
                    if member.size < 0:
                        raise ValueError("GeoIP archive contains a negative file size")
                    expanded_bytes += member.size
                    if expanded_bytes > MAX_EXPANDED_MEMBER_BYTES:
                        raise ValueError(f"GeoIP archive expanded members exceed {MAX_EXPANDED_MEMBER_BYTES} bytes")
                    if path.name != "GeoLite2-City.mmdb":
                        continue
                    candidate_count += 1
                    if candidate_count != 1:
                        raise ValueError("GeoIP archive contains duplicate MMDB entries")
                    if len(path.parts) != 2 or not path.parts[0].startswith("GeoLite2-City_"):
                        raise ValueError("MMDB file is not at the expected archive path")
                    if member.size <= 0 or member.size > MAX_MMDB_BYTES:
                        raise ValueError(f"MMDB database must be non-empty and at most {MAX_MMDB_BYTES} bytes")
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError("MMDB archive entry could not be read")
                    out_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    out_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                    out_fd = os.open(output_path, out_flags, 0o600)
                    output_created = True
                    try:
                        os.fchmod(out_fd, 0o600)
                        with source, os.fdopen(out_fd, "wb") as destination:
                            out_fd = -1
                            while True:
                                chunk = source.read(EXTRACT_CHUNK_BYTES)
                                if not chunk:
                                    break
                                if extracted_bytes + len(chunk) > MAX_MMDB_BYTES:
                                    raise ValueError("MMDB database exceeds the 512 MiB safety limit")
                                destination.write(chunk)
                                extracted_bytes += len(chunk)
                            if extracted_bytes != member.size:
                                raise ValueError("MMDB stream length does not match its tar header")
                            destination.flush()
                            os.fsync(destination.fileno())
                    finally:
                        if out_fd >= 0:
                            os.close(out_fd)
            if candidate_count != 1 or extracted_bytes == 0:
                raise ValueError("archive must contain exactly one expected non-empty GeoLite2-City.mmdb file")
            return extracted_bytes
    except Exception:
        if output_created:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
        raise

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: validate_geoip_archive.py ARCHIVE_PATH OUTPUT_PATH", file=sys.stderr)
        return 2
    try:
        total = validate_and_extract(argv[0], argv[1])
    except Exception as exc:
        print(f"Error: rejected GeoLite2 archive: {exc}", file=sys.stderr)
        return 1
    print(f"Validated {total} bytes of GeoLite2 City database.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
