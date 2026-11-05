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
        if cv2 is None:
            return
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.ok = self.cap.isOpened()
        if self.ok:
            threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while True:
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
        return np.clip((g - lo) / max(hi - lo, 8.0), 0.0, 1.0)

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
