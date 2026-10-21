"""Flask application entry point for News Category Classifier backend."""

import os
import sys
from pathlib import Path

# Add project root and src to sys.path so modules can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import logging

logger = logging.getLogger("newsclf.app")

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from backend.api import api_bp


def create_app() -> Flask:
    """Create and configure the Flask web application."""
    app = Flask(__name__)

    # 16MB max upload size limit for batch CSV security
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

    # Enable Cross-Origin Resource Sharing (CORS) for all frontend clients
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Register API blueprint containing all routes
    app.register_blueprint(api_bp)

    # Global JSON error handlers: never expose stack traces to client
    @app.errorhandler(400)
    def bad_request(e):
        msg = getattr(e, "description", "Bad Request")
        return jsonify({"error": msg}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed."}), 405

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return jsonify({"error": "⚠ Uploaded file exceeds the 16MB maximum file size limit."}), 413

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Internal server error: %s", e)
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

    # Optional convenient route to serve frontend dashboard
    frontend_dir = PROJECT_ROOT / "frontend"

    @app.route("/app/")
    @app.route("/app/<path:filename>")
    def serve_frontend_dashboard(filename="index.html"):
        return send_from_directory(frontend_dir, filename)

    return app



app = create_app()


def main():
    """Run Flask development or production server."""
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    use_prod = os.environ.get("FLASK_ENV") == "production" or "--prod" in sys.argv
    if use_prod:
        try:
            from waitress import serve
            print(f"Starting production Waitress WSGI server on http://{host}:{port}")
            serve(app, host=host, port=port)
            return
        except ImportError:
            pass
    print(f"Starting News Category Classifier Flask API on http://{host}:{port}")
    app.run(host=host, port=port, debug=False if use_prod else True)


if __name__ == "__main__":
    main()
