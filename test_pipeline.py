"""
test_pipeline.py
-----------------
A hands-on demo of the pipeline we just built. Run this directly:

    python3 test_pipeline.py

Walks through:
  1. Registering "person_b" using their selfie photo
  2. Recognizing person_b's OWN photo again -> should MATCH (same person)
  3. Recognizing a DIFFERENT person's photo -> should NOT match

This proves detector -> embedder -> matcher -> storage all work together.
"""

from core.pipeline import register, recognize

SELFIE_PATH = "samples/person_b_selfie.png"
MEME_PATH = "samples/person_a_meme.png"

print("=" * 60)
print("STEP 1: Register person_b using their selfie")
print("=" * 60)
embedding = register("person_b", SELFIE_PATH)
print(f"Registered 'person_b'. Embedding vector length: {len(embedding)}")
print(f"First 5 values of the vector: {embedding[:5]}")

print()
print("=" * 60)
print("STEP 2: Recognize the SAME photo again (expect a MATCH)")
print("=" * 60)
results = recognize(SELFIE_PATH)
for r in results:
    print(f"Found face -> name: {r['name']}, similarity score: {r['score']}, region: {r['region']}")

print()
print("=" * 60)
print("STEP 3: Recognize a DIFFERENT person's photo (expect NO match)")
print("=" * 60)
results = recognize(MEME_PATH)
for r in results:
    print(f"Found face -> name: {r['name']}, similarity score: {r['score']}, region: {r['region']}")
