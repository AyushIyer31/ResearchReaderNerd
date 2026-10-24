"""Where reports, uploads and run records live: a local folder, or an S3 bucket in production.

Keys are slash-separated paths such as "reports/1706.03762.json" or "uploads/<id>.pdf".
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def read_bytes(self, key: str) -> bytes | None: ...
    def write_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...
    def list(self, prefix: str) -> list[tuple[str, float]]:
        """Keys under `prefix`, with their last-modified time (seconds since the epoch)."""
        ...

    def read_text(self, key: str) -> str | None:
        data = self.read_bytes(key)
        return None if data is None else data.decode("utf-8")

    def write_text(self, key: str, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        self.write_bytes(key, text.encode("utf-8"), content_type)


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Key {key!r} points outside the storage folder.")
        return path

    def read_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def write_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def list(self, prefix: str) -> list[tuple[str, float]]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        root = self.root.resolve()
        return [(p.relative_to(root).as_posix(), p.stat().st_mtime) for p in base.rglob("*") if p.is_file()]

    read_text = Storage.read_text
    write_text = Storage.write_text


class S3Storage:
    def __init__(self, bucket: str, prefix: str = "", client=None):
        if client is None:
            import boto3  # only needed in production

            client = boto3.client("s3")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    def read_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def write_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data, ContentType=content_type)

    def list(self, prefix: str) -> list[tuple[str, float]]:
        out = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            for obj in page.get("Contents", []):
                out.append((obj["Key"][len(self.prefix):], obj["LastModified"].timestamp()))
        return out

    read_text = Storage.read_text
    write_text = Storage.write_text
