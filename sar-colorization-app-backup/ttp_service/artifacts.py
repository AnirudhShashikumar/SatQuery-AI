"""Bounded in-memory PNG artifacts addressed only by opaque identifiers."""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from PIL import Image

from .schemas import ArtifactReference


@dataclass
class StoredArtifact:
    data: bytes
    kind: str
    created_at: float


class ArtifactStore:
    def __init__(self) -> None:
        self.max_items = max(8, int(os.getenv("TTP_ARTIFACT_MAX_ITEMS", "128")))
        self.ttl_seconds = max(60, int(os.getenv("TTP_ARTIFACT_TTL_SECONDS", "1800")))
        self._items: "OrderedDict[str, StoredArtifact]" = OrderedDict()
        self._lock = threading.RLock()

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        for artifact_id, item in list(self._items.items()):
            if item.created_at < cutoff:
                self._items.pop(artifact_id, None)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def put_image(self, image: Image.Image, kind: str) -> ArtifactReference:
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        data = output.getvalue()
        artifact_id = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._items[artifact_id] = StoredArtifact(data=data, kind=kind, created_at=time.time())
        return ArtifactReference(
            id=artifact_id,
            kind=kind,
            endpoint=f"/artifacts/{artifact_id}",
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def get(self, artifact_id: str) -> StoredArtifact | None:
        if len(artifact_id) != 32 or any(character not in "0123456789abcdef" for character in artifact_id):
            return None
        with self._lock:
            self._cleanup()
            item = self._items.get(artifact_id)
            if item:
                self._items.move_to_end(artifact_id)
            return item


ARTIFACT_STORE = ArtifactStore()
