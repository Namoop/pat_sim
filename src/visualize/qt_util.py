"""Shared Qt helpers for visualization windows."""

from __future__ import annotations


def configure_qt_platform() -> None:
    """VTK/Qt embedded windows need X11 on many Linux Wayland sessions."""
    import os
    import sys

    if sys.platform.startswith("linux") and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"


def install_sigint_handler(app, window) -> None:
    """Allow Ctrl+C to close the window cleanly while Qt owns the event loop."""
    import signal

    from PyQt6.QtCore import QTimer

    def _handle_sigint(_signum, _frame) -> None:
        window.close()
        app.quit()

    signal.signal(signal.SIGINT, _handle_sigint)
    timer = QTimer(app)
    timer.timeout.connect(lambda: None)
    timer.start(200)
