"""Tests for the URI-aware I/O layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cloud.s3_io import (
    LocalFSClient,
    make_client,
    parse_s3_uri,
    uri_to_local_path,
)


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/key.txt") == ("bucket", "key.txt")
    assert parse_s3_uri("s3://bucket/path/to/key.txt") == ("bucket", "path/to/key.txt")


def test_parse_s3_uri_rejects_non_s3() -> None:
    with pytest.raises(ValueError):
        parse_s3_uri("https://example.com/x")


def test_parse_s3_uri_rejects_missing_bucket() -> None:
    with pytest.raises(ValueError):
        parse_s3_uri("s3:///key.txt")


def test_uri_to_local_path_handles_file_scheme(tmp_path: Path) -> None:
    assert uri_to_local_path("file:///abs/path") == Path("/abs/path")
    assert uri_to_local_path("/abs/path") == Path("/abs/path")
    assert uri_to_local_path("relative/path") == Path("relative/path")
    assert uri_to_local_path("s3://b/k") is None


def test_local_fs_client_round_trips_s3_uris(tmp_path: Path) -> None:
    client = LocalFSClient(root=tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello world")

    client.upload(src, "s3://test-bucket/path/to/dest.bin")
    assert (tmp_path / "test-bucket" / "path" / "to" / "dest.bin").read_bytes() == b"hello world"

    dest = tmp_path / "dest.bin"
    client.download("s3://test-bucket/path/to/dest.bin", dest)
    assert dest.read_bytes() == b"hello world"


def test_local_fs_client_upload_bytes_and_read_text(tmp_path: Path) -> None:
    client = LocalFSClient(root=tmp_path)
    client.upload_bytes(b'{"x":1}', "s3://b/k.json")
    assert client.read_text("s3://b/k.json") == '{"x":1}'


def test_local_fs_client_passes_through_file_paths(tmp_path: Path) -> None:
    """file:// and bare paths bypass the bucket-rewriting."""
    client = LocalFSClient(root=tmp_path / "root")
    direct = tmp_path / "direct.txt"
    direct.write_text("direct content")
    assert client.read_text(str(direct)) == "direct content"


def test_make_client_returns_local_for_file_uri(tmp_path: Path) -> None:
    c = make_client(f"file://{tmp_path}/x.json")
    assert isinstance(c, LocalFSClient)


def test_make_client_returns_local_for_bare_path(tmp_path: Path) -> None:
    c = make_client(str(tmp_path / "x.json"))
    assert isinstance(c, LocalFSClient)
