from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io, cv2, numpy as np
from PIL import Image
from detect import load_haar_face, detect_faces

app = Flask(__name__)
CORS(app)

_face_cascade = load_haar_face()

def apply_mosaic(image, x, y, w, h, factor=20):
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return image
    small = cv2.resize(roi, (max(1, w//factor), max(1, h//factor)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/mosaic/face", methods=["POST"])
def mosaic_face():
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = detect_faces(img_bgr, _face_cascade)
    for (x, y, w, h) in faces:
        img_bgr = apply_mosaic(img_bgr, x, y, w, h)

    _, buf = cv2.imencode(".png", img_bgr)
    return send_file(io.BytesIO(buf), mimetype="image/png")

@app.route("/api/detect/face", methods=["POST"])
def detect_face():
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    faces = detect_faces(img_bgr, _face_cascade)
    return jsonify({"faces": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in faces]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
