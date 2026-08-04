"""
matcher.py
----------
This file answers: "given a new face embedding, whose face does it match
in our known-faces database (if any)?"

THE CORE IDEA: COSINE SIMILARITY
An embedding is just a vector, e.g. [0.12, -0.44, 0.03, ...]. Think of it
as an arrow pointing in some direction in high-dimensional space. Two faces
of the same person point in a very similar direction (small angle between
them), even if the exact numbers differ a bit due to lighting/angle.

Cosine similarity measures the COSINE OF THE ANGLE between two vectors:
  - 1.0  -> vectors point in exactly the same direction (identical face)
  - 0.0  -> vectors are unrelated (90 degrees apart)
  - -1.0 -> vectors point in opposite directions

Formula:  cos(theta) = (A . B) / (|A| * |B|)
  where A . B is the dot product, and |A|, |B| are vector magnitudes.

We compute this between the new embedding and every known embedding, then
pick the best match IF it clears a similarity threshold. The threshold
matters a lot in practice:
  - Too low  -> false positives (mistaking strangers for known people)
  - Too high -> false negatives (rejecting the right person)
For ArcFace, a cosine similarity threshold around 0.68 is a common
starting point.
"""

import numpy as np

SIMILARITY_THRESHOLD = 0.68


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def find_best_match(new_embedding, known_faces: dict):
    """
    known_faces: dict of {name: embedding_vector}
    Returns (name, similarity_score) for the best match above threshold,
    or (None, best_score_found) if nothing clears the threshold.
    """
    best_name = None
    best_score = -1.0  # cosine similarity ranges from -1 to 1

    for name, known_embedding in known_faces.items():
        score = cosine_similarity(new_embedding, np.array(known_embedding))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    else:
        # Best match wasn't confident enough - treat as "unknown person"
        return None, best_score
