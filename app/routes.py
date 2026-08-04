"""
routes.py
---------
The web-facing layer. Three routes:

  GET  /             -> the HTML page with the video feed + register form
  GET  /video_feed    -> the live MJPEG stream (what the <img> tag points to)
  POST /register      -> captures the current frame and saves it under a name

Notice this file knows nothing about DeepFace, embeddings, or cosine
similarity - it only calls camera.get_raw_frame() and pipeline.register().
That separation is the whole point of the core/ layer: the web framework
could be swapped for FastAPI or Django here without touching core/ at all.
"""

from flask import Blueprint, render_template, Response, request, jsonify  # type: ignore[import]

from app.camera import get_camera, generate_mjpeg
from core.pipeline import register

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/video_feed")
def video_feed():
    camera = get_camera()
    return Response(
        generate_mjpeg(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/register", methods=["POST"])
def register_route():
    name = request.form.get("name", "").strip()
    if not name:
        return jsonify({"success": False, "message": "Please enter a name."}), 400

    camera = get_camera()
    frame = camera.get_raw_frame()
    if frame is None:
        return jsonify({"success": False, "message": "Could not read from camera."}), 500

    try:
        register(name, frame)
        return jsonify({"success": True, "message": f"Registered '{name}' successfully."})
    except ValueError:
        # DeepFace raises ValueError when enforce_detection=True finds no face
        return jsonify({
            "success": False,
            "message": "No face detected - center your face in the frame and try again.",
        }), 400
