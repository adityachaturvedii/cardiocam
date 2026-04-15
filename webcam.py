import cv2
import numpy as np
import sys
import time


def list_cameras(max_probe=4):
    """Enumerate reachable camera indices. On macOS prefer AVFoundation so
    multi-camera setups (FaceTime HD + iPhone Continuity Camera) are both
    visible; the default backend often only exposes index 0."""
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    else:
        backend = cv2.CAP_ANY

    found = []
    for idx in range(max_probe):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            found.append((idx, (frame.shape[1], frame.shape[0])))
    return found


class Webcam(object):
    def __init__(self):
        self.dirname = ""  # unused; kept so Webcam and Video share an interface
        self.cap = None
        self.index = 0
        self.valid = False
        self.shape = None

    def set_index(self, idx):
        self.index = int(idx)

    def start(self):
        print("[INFO] Start webcam (index {})".format(self.index))
        time.sleep(1)  # camera warmup
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, backend)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except Exception:
            self.shape = None

    def get_frame(self):
        if self.valid:
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8)
            col = (0, 256, 256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                        (65, 220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("[INFO] Stop webcam")
