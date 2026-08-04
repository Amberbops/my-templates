"""
camera.py
---------
Wraps the webcam and turns it into a live MJPEG stream, with recognition
results drawn directly onto each frame as boxes + name labels.

KEY PERFORMANCE IDEA:
Running the full detect+embed+match pipeline on EVERY single frame would
be far too slow for a smooth video feed (ArcFace inference takes real
time, even on CPU). So instead we only run recognize() every N frames
(RECOGNIZE_EVERY_N_FRAMES), and reuse ("cache") the last known results
to draw boxes on the frames in between. The boxes will lag very slightly
behind fast motion, but the video itself stays smooth.

This is the same tradeoff real-world video AI systems make constantly:
detect/recognize periodically, track or reuse results in between.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
from core.pipeline import recognize

RECOGNIZE_EVERY_N_FRAMES = 15  # tune: lower = more responsive, higher = faster video

# Colors in BGR (OpenCV's format, not RGB)
COLOR_KNOWN = (0, 200, 0)      # green box for a recognized person
COLOR_UNKNOWN = (0, 0, 220)    # red box for an unrecognized face


class VideoCamera:
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
        self.frame_count = 0
        self.last_results = []  # cached recognition results between runs

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_raw_frame(self):
        """Grab a single fresh frame, with no boxes drawn - used for registration."""
        success, frame = self.video.read()
        return frame if success else None

    def get_annotated_jpeg(self):
        """
        Grab a frame, periodically run recognition on it, draw boxes/labels,
        and return it JPEG-encoded (ready to stream to a browser).
        """
        success, frame = self.video.read()
        if not success:
            return None

        self.frame_count += 1
        if self.frame_count % RECOGNIZE_EVERY_N_FRAMES == 0:
            try:
                self.last_results = recognize(frame)
            except Exception:
                # e.g. no face in frame this round - keep showing nothing new
                self.last_results = []

        for result in self.last_results:
            region = result["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            is_known = result["name"] is not None
            color = COLOR_KNOWN if is_known else COLOR_UNKNOWN
            label = f"{result['name']} ({result['score']:.2f})" if is_known else "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        ok, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes() if ok else None


_camera_instance = None


def get_camera():
    """Lazily create a single shared camera instance the whole app reuses."""
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = VideoCamera()
    return _camera_instance


def generate_mjpeg(camera: VideoCamera):
    """
    A generator that yields frames in the 'multipart/x-mixed-replace' format
    browsers understand as a live video stream (MJPEG).
    """
    while True:
        jpeg_bytes = camera.get_annotated_jpeg()
        if jpeg_bytes is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
        )
