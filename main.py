import os
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from ui.mainwindow import MainWindow


def _prefer_glx_on_wayland():
    """moderngl attaches to Qt's GL context through GLX only. Qt on Wayland uses EGL,
    where that detection is unreliable: the preview happened to work, a second GL
    window (the projector output) never initialised. Run through XWayland instead,
    unless the user chose a platform."""
    if (sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY")
            and os.environ.get("DISPLAY") and "QT_QPA_PLATFORM" not in os.environ):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_glx")


def _core_profile_on_macos():
    """macOS hands out a legacy 2.1 context unless a core profile is requested, and
    the shaders are #version 330 core. 4.1 is the highest macOS supports."""
    if sys.platform == "darwin":
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)


def main():
    _prefer_glx_on_wayland()
    _core_profile_on_macos()
    QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Visualizer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
