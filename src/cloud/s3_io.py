"""URI-aware I/O so the handler runs identically locally and in S3.

The handler accepts ``video_uri`` and ``camera_track_uri`` that may be
either ``s3://bucket/key`` or ``file:///abs/path`` (or a bare path,
which is interpreted as a file path). The :class:`S3Client` Protocol
covers both — tests inject :class:`LocalFSClient` to avoid boto3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, key) for an ``s3://...`` URI."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"not an s3:// URI: {uri!r}")
    if not parsed.netloc:
        raise ValueError(f"s3 URI missing bucket: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def uri_to_local_path(uri: str) -> Path | None:
    """If ``uri`` is a local file (``file://`` or bare path), return it.

    Returns ``None`` for ``s3://`` URIs so callers can dispatch.
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        # file:///abs/path or bare path
        path = parsed.path if parsed.scheme == "file" else uri
        return Path(path)
    return None


class S3Client(Protocol):
    """Smallest surface the handler needs."""

    def download(self, uri: str, dest: Path) -> None: ...

    def upload(self, src: Path, uri: str) -> None: ...

    def upload_bytes(self, data: bytes, uri: str) -> None: ...

    def read_text(self, uri: str) -> str: ...


class LocalFSClient:
    """File-system implementation. ``s3://bucket/key`` is rewritten to
    ``<root>/bucket/key`` so multiple "buckets" can coexist in a tmpdir
    during tests; ``file://`` and bare paths use the FS directly.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _resolve(self, uri: str) -> Path:
        local = uri_to_local_path(uri)
        if local is not None:
            return local
        bucket, key = parse_s3_uri(uri)
        return self.root / bucket / key

    def download(self, uri: str, dest: Path) -> None:
        src = self._resolve(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())

    def upload(self, src: Path, uri: str) -> None:
        dest = self._resolve(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(src).read_bytes())

    def upload_bytes(self, data: bytes, uri: str) -> None:
        dest = self._resolve(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def read_text(self, uri: str) -> str:
        return self._resolve(uri).read_text()


class Boto3S3Client:
    """Real-S3 implementation, used by the container at runtime."""

    def __init__(self, region: str | None = None) -> None:
        import boto3

        self._s3 = boto3.client("s3", region_name=region)

    def download(self, uri: str, dest: Path) -> None:
        bucket, key = parse_s3_uri(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(bucket, key, str(dest))

    def upload(self, src: Path, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        self._s3.upload_file(str(src), bucket, key)

    def upload_bytes(self, data: bytes, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        self._s3.put_object(Bucket=bucket, Key=key, Body=data)

    def read_text(self, uri: str) -> str:
        bucket, key = parse_s3_uri(uri)
        obj = self._s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")


def make_client(uri: str) -> S3Client:
    """Pick the right client for the protocol used by ``uri``.

    The handler doesn't know in advance whether it's running locally
    (manifest path comes from the CLI) or in a Batch container (manifest
    URI is an ``s3://``).
    """
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return Boto3S3Client()
    return LocalFSClient(root=Path("/"))
