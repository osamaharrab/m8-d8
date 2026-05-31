"""Module 8 — Core Skills Drill: RAG Basics.

Three operational primitives for RAG: embed a sentence, verify a Weaviate
connection, ingest a small set of objects with externally-supplied vectors.

Submit by branching `drill-8-rag-basics`, opening a PR, pasting the PR URL
into TalentLMS → Module 8 → Core Skills Drill.
"""

import numpy as np
import weaviate

_model = None


def embed_text(text: str) -> np.ndarray:
    """Return a 384-dim float32 numpy vector for the input string.

    Use sentence-transformers' all-MiniLM-L6-v2.

    Hint:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        v = model.encode(text, convert_to_numpy=True).astype(np.float32)
    """
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("all-MiniLM-L6-v2")

    vector = _model.encode(text, convert_to_numpy=True)
    return vector.astype(np.float32)


def weaviate_ready(url: str) -> bool:
    """Return True if Weaviate at `url` is reachable and ready, else False.

    Wrap in try/except so a non-running Weaviate returns False rather than
    raising a connection error.
    """
    try:
        return bool(weaviate.Client(url).is_ready())
    except Exception:
        return False


def ingest_corpus(client: weaviate.Client, class_name: str, items: list[dict]) -> int:
    """Ingest items into the named class. Return the count of ingested objects.

    Each item is {"title": str, "text": str, "vector": list[float]}.

    If the class does not exist, create it with:
      - properties: title (text), text (text, BM25-indexed)
      - vectorizer: "none"

    Use client.batch (or with client.batch as batch:) and remember to flush.
    Verify the count via:
      client.query.aggregate(class_name).with_meta_count().do()
    """
    if not client.schema.exists(class_name):
        client.schema.create_class(
            {
                "class": class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "title", "dataType": ["text"]},
                    {"name": "text", "dataType": ["text"]},
                ],
            }
        )

    with client.batch as batch:
        for item in items:
            batch.add_data_object(
                {"title": item["title"], "text": item["text"]},
                class_name=class_name,
                vector=item["vector"],
            )

    result = client.query.aggregate(class_name).with_meta_count().do()
    return int(result["data"]["Aggregate"][class_name][0]["meta"]["count"])
