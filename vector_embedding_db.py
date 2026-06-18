import uuid

import chromadb
from chromadb.config import Settings
import posthog

_original_posthog_capture = posthog.capture

def _posthog_capture(*args, **kwargs):
    if len(args) == 1:
        return _original_posthog_capture(args[0], **kwargs)
    if len(args) == 2:
        return _original_posthog_capture(args[1], distinct_id=args[0], **kwargs)
    if len(args) == 3:
        return _original_posthog_capture(args[1], distinct_id=args[0], properties=args[2], **kwargs)
    return _original_posthog_capture(*args, **kwargs)

posthog.capture = _posthog_capture

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_db",
        is_persistent=True,
        anonymized_telemetry=False,
    )
)

collection = client.get_or_create_collection(
    name="resume_vectors",
    metadata={"hnsw:space": "cosine"},
)


def save_vector(embedding_vector, metadata=None, id=None):
    """Save only the embedding vector into ChromaDB."""
    if id is None:
        id = str(uuid.uuid4())

    ids = [id]
    metadatas = [metadata] if metadata is not None else None

    return collection.add(
        ids=ids,
        embeddings=[embedding_vector],
        metadatas=metadatas,
        documents=[None],
    )


def persist():
    """Persist ChromaDB storage to disk.

    Current ChromaDB versions with file-backed storage persist automatically,
    so no explicit client.persist() method is required.
    """
    if hasattr(client, "persist"):
        client.persist()
    return None

