"""
pipeline.py
-----------
The orchestrator. This is the ONLY file that the rest of an app (Flask
routes, CLI, etc.) needs to talk to. It hides the details of which
embedding model or storage backend is used behind two simple functions:

    register(name, image) -> saves a new known face
    recognize(image)      -> returns [(name_or_None, box, score), ...]
                              for every face found in the image

This is the "glue" layer mentioned earlier -- swap embedder.py's model,
or storage.py's backend, and pipeline.py (and everything above it)
doesn't need to change.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.embedder import get_embedding, get_all_embeddings
from core.matcher import find_best_match
from storage.storage import load_known_faces, register_face


def register(name: str, image_path_or_array):
    """
    Detect the (single) face in the given image, compute its embedding,
    and store it under `name` in the known-faces database.
    """
    embedding = get_embedding(image_path_or_array)
    register_face(name, embedding)
    return embedding


def recognize(image_path_or_array):
    """
    Detect ALL faces in the given image/frame, and for each one, find
    the best matching known name (or None if unrecognized).

    Returns a list of dicts:
        [{"name": str or None, "score": float, "region": {...}}, ...]
    """
    known_faces = load_known_faces()
    faces = get_all_embeddings(image_path_or_array)

    results = []
    for face in faces:
        name, score = find_best_match(face["embedding"], known_faces)
        results.append({
            "name": name,
            "score": round(float(score), 4),
            "region": face["region"],
        })
    return results
