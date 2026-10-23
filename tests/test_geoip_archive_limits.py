"""Resource and archive-boundary regression tests for GeoIP updates."""
import io
import tarfile
from scripts import download_geoip_archive as downloader
from scripts import validate_geoip_archive as validator
import pytest

class FakeResponse:
    def __init__(self, chunks=(), headers=None, status_error=None):
        self.headers = headers or {}
        self.chunks = list(chunks)
        self.status_error = status_error
        self.closed = False
        self.iterated = False
    def raise_for_status(self):
        if self.status_error:
            raise self.status_error
    def iter_content(self, chunk_size):
        self.iterated = True
        self.chunk_size = chunk_size
        return iter(self.chunks)
    def close(self):
        self.closed = True

def _make_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for member, payload in entries:
            if payload is None:
                archive.addfile(member)
            else:
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))

def _mmdb_member(name="GeoLite2-City_20261023/GeoLite2-City.mmdb", payload=b"mmdb-data"):
    member = tarfile.TarInfo(name)
    member.mode = 0o644
    return member, payload

def test_download_streams_auth_and_writes_private_archive(tmp_path, monkeypatch):
    output = tmp_path / "download.tar.gz"
    response = FakeResponse([b"ab", b"cde"], {"Content-Length": "5"})
    calls = []
    def fake_get(url, auth, timeout, stream):
        calls.append((url, auth, timeout, stream))
        return response
    monkeypatch.setattr(downloader.requests, "get", fake_get)
    total = downloader.download_archive(output, account_id="12345", license_key="abcDEF123")
    assert total == 5
    assert output.read_bytes() == b"abcde"
    assert output.stat().st_mode & 0o777 == 0o600
    assert calls == [(downloader.GEOIP_DOWNLOAD_URL, ("12345", "abcDEF123"), (10, 60), True)]
    assert response.chunk_size == downloader.DOWNLOAD_CHUNK_BYTES
    assert response.closed

def test_download_rejects_declared_oversize_before_creating_output(tmp_path, monkeypatch):
    output = tmp_path / "download.tar.gz"
    response = FakeResponse([b"must-not-read"], {"Content-Length": str(downloader.MAX_ARCHIVE_BYTES + 1)})
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="download limit"):
        downloader.download_archive(output, account_id="123", license_key="key123")
    assert not output.exists()
    assert response.iterated is False
    assert response.closed

def test_download_enforces_streamed_limit_and_removes_partial_file(tmp_path, monkeypatch):
    output = tmp_path / "download.tar.gz"
    response = FakeResponse([b"1234", b"5678"], {})
    monkeypatch.setattr(downloader, "MAX_ARCHIVE_BYTES", 6)
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="download limit"):
        downloader.download_archive(output, account_id="123", license_key="key123")
    assert not output.exists()
    assert response.closed

def test_download_does_not_overwrite_existing_file_or_follow_symlink(tmp_path, monkeypatch):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    output = tmp_path / "download.tar.gz"
    output.symlink_to(victim)
    response = FakeResponse([b"replacement"], {})
    monkeypatch.setattr(downloader.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(FileExistsError):
        downloader.download_archive(output, account_id="123", license_key="key123")
    assert victim.read_text(encoding="utf-8") == "keep"
    assert output.is_symlink()
    assert response.closed

def test_validator_streams_and_extracts_only_expected_mmdb(tmp_path):
    archive_path = tmp_path / "valid.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    directory = tarfile.TarInfo("GeoLite2-City_20261023/")
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o755
    payload = b"test-mmdb-payload"
    _make_archive(archive_path, [(directory, None), _mmdb_member(payload=payload)])
    total = validator.validate_and_extract(archive_path, output)
    assert total == len(payload)
    assert output.read_bytes() == payload
    assert output.stat().st_mode & 0o777 == 0o600

def test_validator_rejects_nonregular_member_and_cleans_output(tmp_path):
    archive_path = tmp_path / "link.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    link = tarfile.TarInfo("GeoLite2-City_20261023/payload-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    _make_archive(archive_path, [_mmdb_member(), (link, None)])
    with pytest.raises(ValueError, match="link or special file"):
        validator.validate_and_extract(archive_path, output)
    assert not output.exists()

def test_validator_rejects_parent_traversal_member(tmp_path):
    archive_path = tmp_path / "traversal.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    _make_archive(archive_path, [(tarfile.TarInfo("../escaped"), b"bad"), _mmdb_member()])
    with pytest.raises(ValueError, match="parent-traversal"):
        validator.validate_and_extract(archive_path, output)
    assert not output.exists()

def test_validator_rejects_duplicate_expected_mmdb_entries(tmp_path):
    archive_path = tmp_path / "duplicate.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    _make_archive(archive_path, [_mmdb_member(), _mmdb_member(payload=b"second")])
    with pytest.raises(ValueError, match="duplicate MMDB"):
        validator.validate_and_extract(archive_path, output)
    assert not output.exists()

def test_validator_caps_member_count(tmp_path, monkeypatch):
    archive_path = tmp_path / "many.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    entries = [(tarfile.TarInfo("GeoLite2-City_20261023/"), None), _mmdb_member(), (tarfile.TarInfo("notes.txt"), b"notes")]
    _make_archive(archive_path, entries)
    monkeypatch.setattr(validator, "MAX_ARCHIVE_MEMBERS", 2)
    with pytest.raises(ValueError, match="member limit"):
        validator.validate_and_extract(archive_path, output)
    assert not output.exists()

def test_validator_rejects_archive_size_before_parsing(tmp_path, monkeypatch):
    archive_path = tmp_path / "large.tar.gz"
    output = tmp_path / "GeoLite2-City.mmdb"
    _make_archive(archive_path, [_mmdb_member()])
    monkeypatch.setattr(validator, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="archive must be non-empty"):
        validator.validate_and_extract(archive_path, output)
    assert not output.exists()
