"""The webcam, for the mirror effects: the newest frame from a background thread, as brightness or motion.

Nothing leaves the laptop and nothing is stored: frames are only turned into masks for the effects.
"""

import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # the effects fall back to running without a camera
    cv2 = None


class Camera:
    def __init__(self, index=0):
        self.frame = None
        self.lock = threading.Lock()
        self.ok = False
        self.prev = {}
        self.running = False
        self.thread = None
        self.contrast = 0.0
        if cv2 is None:
            return
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.ok = self.cap.isOpened()
        if self.ok:
            self.running = True
            self.thread = threading.Thread(target=self._grab, daemon=True)
            self.thread.start()

    def close(self):
        """Stop the reader thread and let go of the camera. Without this, quitting while the thread is
        inside OpenCV's read() makes the C++ runtime abort the whole program."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        if cv2 is not None and getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None
        self.ok = False

    def _grab(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            with self.lock:
                self.frame = g

    def gray(self, w, h):
        """The newest frame as brightness 0..1, mirrored, cropped to w:h and scaled to w x h; stretched
        so the 5th..95th percentile spans 0..1, which keeps a dim room usable. None without a camera."""
        with self.lock:
            f = self.frame
        if f is None:
            return None
        fh, fw = f.shape
        if fw / fh > w / h:  # crop to the world's shape
            cw = int(fh * w / h)
            f = f[:, (fw - cw) // 2:(fw + cw) // 2]
        else:
            ch = int(fw * h / w)
            f = f[(fh - ch) // 2:(fh + ch) // 2]
        g = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)[:, ::-1].astype(np.float32)
        lo, hi = np.percentile(g[::4, ::4], (5, 95))
        self.contrast = float(hi - lo)
        # stretch a dim picture, but not sensor noise: under 40 grey levels of contrast it stays dim
        return np.clip((g - lo) / max(hi - lo, 40.0), 0.0, 1.0)

    def motion(self, w, h, key="default"):
        """How much each spot changed since the last call with this key: 0..1. None without a camera."""
        g = self.gray(w, h)
        if g is None:
            return None
        prev = self.prev.get(key)
        self.prev[key] = g
        if prev is None or prev.shape != g.shape:
            return np.zeros_like(g)
        d = np.abs(g - prev)
        return np.clip(cv2.GaussianBlur(d, (0, 0), 1.5) * 4.0, 0.0, 1.0)
