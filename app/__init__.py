"""
__init__.py (app factory)
--------------------------
Using an "app factory" function rather than a bare `app = Flask(__name__)`
at import time is a common Flask pattern - it makes the app easier to
test (you can create fresh instances) and avoids import-order issues.
"""

import os
from flask import Flask


def create_app():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, "templates"),
        static_folder=os.path.join(project_root, "static"),
    )

    from app.routes import bp
    app.register_blueprint(bp)

    return app
