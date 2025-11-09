import os, cv2
from typing import Tuple

def load_haar_face():
    candidates = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        os.path.join(os.path.dirname(cv2.__file__), "cv2", "data", "haarcascade_frontalface_default.xml"),
    ]
    tried = []
    for p in candidates:
        if p and os.path.exists(p):
            face_cascade = cv2.CascadeClassifier(p)
            if not face_cascade.empty():
                return face_cascade
            tried.append(p + " (empty)")
        else:
            tried.append(p + " (missing)")
    raise RuntimeError("Haar cascade 로드 실패. 확인 경로: " + " | ".join(tried))

def detect_faces(frame_gray, face_cascade, scale_factor: float, min_neighbors: int, min_size: int):
    rects = face_cascade.detectMultiScale(
        frame_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_size, min_size)
    )
    return rects
