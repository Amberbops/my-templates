"""
storage.py
----------
Persists the "known faces" database to disk as a pickle file, so
registered faces survive between app restarts.

Structure on disk: a dict of {name: embedding_as_list}
We store as plain Python lists (not numpy arrays) inside the pickle so
the file stays portable; matcher.py converts back to numpy when needed.

This is intentionally simple for learning. In a real production system
you'd swap this file for a proper database (SQLite, Postgres, or a
vector database like FAISS/Pinecone for large-scale search) -- but the
rest of the app (embedder.py, matcher.py, pipeline.py) wouldn't need to
change, because they only depend on getting a `dict[name] = embedding`.
"""

import pickle
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "faces_db.pkl")


def load_known_faces() -> dict:
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def save_known_faces(known_faces: dict):
    with open(DB_PATH, "wb") as f:
        pickle.dump(known_faces, f)


def register_face(name: str, embedding):
    known_faces = load_known_faces()
    known_faces[name] = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    save_known_faces(known_faces)


def delete_face(name: str):
    known_faces = load_known_faces()
    if name in known_faces:
        del known_faces[name]
        save_known_faces(known_faces)
