"""SASLite Desktop Client — pywebview wrapper for Flask app."""

from __future__ import annotations

import threading
import time

import webview

from saslite.gui.app import app


def start_flask():
    """Start Flask server in background thread."""
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


def main() -> int:
    """Launch desktop GUI."""
    # Start Flask in background
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Wait for Flask to start
    time.sleep(1)

    # Create desktop window
    window = webview.create_window(
        title="SASLite Desktop",
        url="http://127.0.0.1:5000",
        width=1200,
        height=800,
        resizable=True,
        fullscreen=False,
    )

    # Start GUI event loop
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
