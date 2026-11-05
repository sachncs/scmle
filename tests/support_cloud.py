"""In-memory S3 and GCS clients implementing the protocols the backends use."""

# ruff: noqa: N803, N818  (argument and error names follow the SDKs)

import datetime
import io
import threading


class FakeClientError(Exception):
    """Shaped like botocore's ClientError."""

    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    """A bucket in a dict, with conditional puts, pages and fault injection."""

    def __init__(self, page_size=10):
        self.objects = {}
        self.page_size = page_size
        self.lock = threading.Lock()
        self.faults = []
        self.calls = []

    def fail_next(self, *codes):
        self.faults.extend(codes)

    def maybe_fail(self, name):
        self.calls.append(name)
        if self.faults:
            raise FakeClientError(self.faults.pop(0))

    def put_object(self, Bucket, Key, Body, Metadata=None, IfNoneMatch=None):
        self.maybe_fail("put")
        with self.lock:
            if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
                raise FakeClientError("PreconditionFailed")
            self.objects[(Bucket, Key)] = {
                "data": bytes(Body),
                "meta": dict(Metadata or {}),
                "at": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
            }
        return {}

    def entry(self, Bucket, Key):
        found = self.objects.get((Bucket, Key))
        if found is None:
            raise FakeClientError("NoSuchKey")
        return found

    def get_object(self, Bucket, Key):
        self.maybe_fail("get")
        found = self.entry(Bucket, Key)
        return {
            "Body": io.BytesIO(found["data"]),
            "Metadata": found["meta"],
            "LastModified": found["at"],
            "ContentLength": len(found["data"]),
        }

    def head_object(self, Bucket, Key):
        self.maybe_fail("head")
        found = self.entry(Bucket, Key)
        return {
            "ContentLength": len(found["data"]),
            "Metadata": found["meta"],
            "LastModified": found["at"],
        }

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        self.maybe_fail("list")
        keys = sorted(
            k for b, k in self.objects if b == Bucket and k.startswith(Prefix)
        )
        start = int(ContinuationToken or 0)
        page = keys[start : start + self.page_size]
        more = start + self.page_size < len(keys)
        reply = {"Contents": [{"Key": k} for k in page], "IsTruncated": more}
        if more:
            reply["NextContinuationToken"] = str(start + self.page_size)
        return reply

    def delete_object(self, Bucket, Key):
        self.maybe_fail("delete")
        with self.lock:
            self.objects.pop((Bucket, Key), None)
        return {}


class FakeNotFound(Exception):
    code = 404


class FakePreconditionFailed(Exception):
    code = 412


class FakeServiceUnavailable(Exception):
    code = 503


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.metadata = None
        self.size = None
        self.updated = None
        self.generation = None

    def upload_from_string(
        self, data, content_type=None, if_generation_match=None
    ):
        self.bucket.client.maybe_fail("put")
        with self.bucket.client.lock:
            exists = self.name in self.bucket.store
            if if_generation_match == 0 and exists:
                raise FakePreconditionFailed()
            generation = (
                self.bucket.store[self.name]["generation"] + 1 if exists else 1
            )
            self.bucket.store[self.name] = {
                "data": bytes(data),
                "meta": dict(self.metadata or {}),
                "generation": generation,
            }

    def fill(self):
        found = self.bucket.store[self.name]
        self.metadata = dict(found["meta"])
        self.size = len(found["data"])
        self.generation = found["generation"]
        self.updated = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

    def download_as_bytes(self):
        self.bucket.client.maybe_fail("get")
        if self.name not in self.bucket.store:
            raise FakeNotFound()
        return self.bucket.store[self.name]["data"]

    def delete(self):
        self.bucket.client.maybe_fail("delete")
        with self.bucket.client.lock:
            if self.name not in self.bucket.store:
                raise FakeNotFound()
            del self.bucket.store[self.name]


class FakeBucket:
    def __init__(self, client, name):
        self.client = client
        self.store = client.buckets.setdefault(name, {})

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        self.client.maybe_fail("head")
        if name not in self.store:
            return None
        blob = FakeBlob(self, name)
        blob.fill()
        return blob

    def list_blobs(self, prefix=""):
        self.client.maybe_fail("list")
        for name in sorted(self.store):
            if name.startswith(prefix):
                blob = FakeBlob(self, name)
                blob.fill()
                yield blob


class FakeGcsClient:
    def __init__(self):
        self.buckets = {}
        self.lock = threading.Lock()
        self.faults = []
        self.calls = []

    def fail_next(self, *errors):
        self.faults.extend(errors)

    def maybe_fail(self, name):
        self.calls.append(name)
        if self.faults:
            raise self.faults.pop(0)

    def bucket(self, name):
        return FakeBucket(self, name)
