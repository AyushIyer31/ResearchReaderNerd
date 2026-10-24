import io
from datetime import datetime, timezone

import pytest

from researchreadernerd.storage import LocalStorage, S3Storage


class FakeS3:
    """Just enough of boto3's S3 client for S3Storage."""

    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects = {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body

    def get_paginator(self, name):
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                when = datetime(2026, 10, 24, tzinfo=timezone.utc)
                yield {"Contents": [{"Key": k, "LastModified": when} for (b, k) in fake.objects if b == Bucket and k.startswith(Prefix)]}

        return Paginator()


@pytest.fixture(params=["local", "s3"])
def storage(request, tmp_path):
    if request.param == "local":
        return LocalStorage(tmp_path / "store")
    return S3Storage("bucket", "rrn", client=FakeS3())


def test_round_trip_and_listing(storage):
    assert storage.read_text("a.json") is None
    storage.write_text("a.json", "{}")
    storage.write_bytes("uploads/x.pdf", b"%PDF")
    assert storage.read_text("a.json") == "{}"
    assert storage.read_bytes("uploads/x.pdf") == b"%PDF"
    assert sorted(k for k, _ in storage.list("")) == ["a.json", "uploads/x.pdf"]
    assert [k for k, _ in storage.list("uploads/")] == ["uploads/x.pdf"]
    assert storage.list("missing/") == []


def test_local_storage_refuses_paths_outside_its_folder(tmp_path):
    with pytest.raises(ValueError):
        LocalStorage(tmp_path / "store").write_text("../escape.txt", "x")
