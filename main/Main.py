from PySide6.QtWidgets import QApplication
from gui.MainWindow import MainWindow
import sys
import signal

def main():
    app = QApplication(sys.argv)
    with open("gui/style.qss", "r") as f:
        app.setStyleSheet(f.read())

    # window
    window = MainWindow()
    window.show()

    def cleanup():
        try:
            if hasattr(window, "worker_thread") and window.worker_thread is not None:
                if hasattr(window.worker_thread, "stop"):
                    window.worker_thread.stop()
            if hasattr(window, "mic") and window.mic is not None:
                if hasattr(window.mic, "stop"):
                    window.mic.stop()
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")

        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
