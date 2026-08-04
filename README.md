# Face Recognition Template

A minimal, reusable face recognition pipeline built around DeepFace (ArcFace
embeddings) with pickle-based storage. Designed to be copied into new
projects — the `core/` and `storage/` folders rarely need to change; only
the app layer on top (Flask routes, CLI, etc.) does.

## How it works (the pipeline)

```
image/frame
    │
    ▼
DeepFace.extract_faces()   -> finds face(s), crops/aligns them
    │
    ▼
DeepFace.represent()       -> turns each face into a 512-number vector
    │                          (core/embedder.py)
    ▼
cosine_similarity()        -> compares new vector against known vectors
    │                          (core/matcher.py)
    ▼
best match above threshold -> name, or None if unrecognized
```

Known faces are stored in `storage/faces_db.pkl` as `{name: embedding_list}`.

## Project structure

```
face-template/
├── core/
│   ├── embedder.py     # image -> face embedding vector (DeepFace/ArcFace)
│   ├── matcher.py       # cosine similarity matching against known faces
│   └── pipeline.py      # orchestrates: register() and recognize()
├── storage/
│   ├── storage.py       # pickle load/save for known faces
│   └── faces_db.pkl     # the known-faces database (created at runtime)
├── samples/              # test images used during development
├── test_pipeline.py      # standalone script proving the pipeline works
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run will download the ArcFace model weights (~140MB) automatically
to `~/.deepface/weights/`.

## Try it

```bash
cd face-template
python3 test_pipeline.py
```

This registers a sample face and then tries to recognize it in two test
images — one matching, one not — printing similarity scores for each.

## Using it in your own code

```python
from core.pipeline import register, recognize

# Register a known person
register("alice", "path/to/alice_photo.jpg")

# Recognize faces in a new image (or a webcam frame / numpy array)
results = recognize("path/to/new_photo.jpg")
for r in results:
    print(r["name"], r["score"], r["region"])
```

`recognize()` returns a list because an image can contain multiple faces:

```python
[{"name": "alice", "score": 0.81, "region": {"x": 94, "y": 73, "w": 266, "h": 266, ...}}]
```

`name` is `None` when no known face clears the similarity threshold —
i.e. "unrecognized person."

## Tuning

- **Similarity threshold** (`core/matcher.py`, `SIMILARITY_THRESHOLD = 0.68`):
  raise it to reduce false positives (stricter matching), lower it to
  reduce false negatives (more lenient matching).
- **Embedding model** (`core/embedder.py`, `MODEL_NAME`): DeepFace supports
  swapping in `"Facenet512"`, `"VGG-Face"`, `"SFace"`, etc. Different
  models have different accuracy/speed tradeoffs and slightly different
  ideal thresholds.
- **Detector backend** (`core/embedder.py`, `DETECTOR_BACKEND`): `"opencv"`
  is fastest; `"retinaface"` or `"mtcnn"` are more accurate but slower —
  worth it for images with difficult angles/lighting.

## Live webcam app (Flask)

```
face-template/
├── app/
│   ├── __init__.py      # Flask app factory
│   ├── camera.py         # webcam capture, recognition overlay, MJPEG stream
│   └── routes.py         # /, /video_feed, /register
├── templates/
│   └── index.html        # video feed + "register current face" form
└── run.py                # entry point
```

Run it:

```bash
python3 run.py
```

Then open **http://127.0.0.1:5000** in your browser. You'll see your
live webcam feed with green boxes around recognized faces (labeled with
name + similarity score) and red boxes around unknown faces. Type a name
and click "Register current face" to add whoever's currently in frame
to the known-faces database — they should show up recognized (green
box) within a second or two.

**Performance note:** running the full recognition pipeline on every
single video frame would make the stream choppy, so `camera.py` only
runs `recognize()` every `RECOGNIZE_EVERY_N_FRAMES` frames (default 15)
and reuses the last result in between. Lower that number for more
responsive boxes at the cost of a slower stream; raise it for smoother
video with slightly laggier labels.

## Next steps (not yet built)

- Multiple camera support / selecting a specific webcam device
- A "delete registered face" button in the UI (the backend function
  `storage.delete_face()` already exists, just not wired to a route)
- Swapping `storage/storage.py` for SQLite or a vector database if the
  known-faces list grows large (pickle is fine for dozens/hundreds of
  faces, but doesn't scale well past that)
- Face tracking between recognition runs (e.g. with OpenCV's built-in
  trackers) so boxes follow motion smoothly between the periodic
  recognition passes, instead of staying static
