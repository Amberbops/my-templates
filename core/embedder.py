"""
embedder.py
-----------
This file answers one question: "given a face image, what is its numeric
fingerprint?"

WHY THIS MATTERS:
A neural network (here, an ArcFace/Facenet-style model loaded by DeepFace)
has been trained on millions of face pairs to produce a vector (a list of
numbers, e.g. 128 or 512 floats) such that:
  - Two photos of the SAME person produce vectors that are close together
  - Two photos of DIFFERENT people produce vectors that are far apart

This vector is called an "embedding". We never compare raw pixels between
two photos (lighting, angle, and background would ruin that comparison).
Instead we compare these embeddings, which the model has learned to make
robust to lighting/angle/expression changes.

DeepFace.represent() does two things internally:
  1. Detects the face and crops/aligns it
  2. Runs the crop through the embedding model
We use its output vector directly.
"""

from deepface import DeepFace
import numpy as np

# The embedding model to use. DeepFace supports several interchangeably:
# "Facenet512", "ArcFace", "VGG-Face", "SFace", etc.
# ArcFace is a strong, widely-used choice (512-dimensional vectors).
MODEL_NAME = "ArcFace"

# The detector DeepFace uses internally to find/crop the face before
# embedding. "opencv" is fastest (good for learning); "retinaface" or
# "mtcnn" are more accurate but slower.
DETECTOR_BACKEND = "opencv"


def get_embedding(image_path_or_array):
    """
    Given a path to an image (or a numpy array frame from a webcam),
    return a single face embedding as a numpy array.

    Raises a ValueError if no face is found.
    """
    result = DeepFace.represent(
        img_path=image_path_or_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,  # raise an error if no face is found
    )

    # DeepFace.represent returns a list because an image could contain
    # multiple faces. We take the first one for this simple case.
    embedding = result[0]["embedding"]
    return np.array(embedding)


def get_all_embeddings(image_path_or_array):
    """
    Like get_embedding, but returns embeddings + bounding boxes for
    EVERY face found in the image. This is what we'll use for the
    live webcam case, where multiple people might be in frame.

    Returns a list of dicts: [{"embedding": ndarray, "region": {...}}, ...]
    """
    results = DeepFace.represent(
        img_path=image_path_or_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,  # don't crash if zero faces; just return []
    )
    faces = []
    for r in results:
        faces.append({
            "embedding": np.array(r["embedding"]),
            "region": r["facial_area"],  # {'x':.., 'y':.., 'w':.., 'h':..}
        })
    return faces
