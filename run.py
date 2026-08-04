"""
run.py
------
Entry point. Run this from the project root:

    python3 run.py

Then open http://127.0.0.1:5000 in your browser. Your webcam permission
prompt may appear depending on your OS/browser security settings for
local apps (usually none needed since OpenCV accesses the camera
directly at the OS level, not via browser getUserMedia).
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True matters here: it lets Flask handle the /register
    # request WHILE the /video_feed stream is still being read by the
    # browser in the background.
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
