from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io, os, cv2, numpy as np, time
from PIL import Image
from deepface import DeepFace
from detect import load_haar_face, detect_faces

app = Flask(__name__)
CORS(app)

# ------------------------------
# 기본 설정
# ------------------------------
FACE_DIR = "detected_face"
os.makedirs(FACE_DIR, exist_ok=True)

MODEL_NAME = "VGG-Face"
COSINE_THRESHOLD = 0.40  # 임계값: 낮을수록 엄격
_face_cascade = load_haar_face()
_model_loaded = False
_face_db_cache = None


def ensure_model():
    global _model_loaded
    if not _model_loaded:
        DeepFace.build_model(MODEL_NAME)
        _model_loaded = True


# ------------------------------
# 유틸 함수
# ------------------------------
def cv_imread_from_upload(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def apply_mosaic(image, x, y, w, h, factor=20):
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return image
    small = cv2.resize(roi, (max(1, w//factor), max(1, h//factor)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image


def cosine_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ------------------------------
# 얼굴 DB 캐시 로딩
# ------------------------------
def rebuild_face_db():
    """detected_face 폴더의 모든 얼굴 이미지를 Embedding하여 캐시"""
    ensure_model()
    db = []
    for fname in os.listdir(FACE_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(FACE_DIR, fname)
        try:
            rep = DeepFace.represent(img_path=path, model_name=MODEL_NAME, enforce_detection=False)
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
            name = fname.split("_")[0]
            db.append((name, np.array(emb)))
        except Exception as e:
            print(f"[WARN] embedding 실패: {fname} ({e})")
    return db


def get_face_db():
    global _face_db_cache
    if _face_db_cache is None:
        _face_db_cache = rebuild_face_db()
    return _face_db_cache


# ------------------------------
# API: 헬스체크
# ------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "faces": len(os.listdir(FACE_DIR))})


# ------------------------------
# API: 얼굴 등록
# ------------------------------
@app.route("/api/face/register", methods=["POST"])
def register_face():
    if "file" not in request.files:
        return jsonify({"error": "file not found"}), 400
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "name missing"}), 400

    f = request.files["file"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    path = os.path.join(FACE_DIR, f"{safe_name}_{ts}.jpg")
    img = cv_imread_from_upload(f)
    cv2.imwrite(path, img)

    # DB 캐시 무효화
    global _face_db_cache
    _face_db_cache = None

    return jsonify({"ok": True, "saved": os.path.basename(path)})


# ------------------------------
# API: 얼굴 목록 조회
# ------------------------------
@app.route("/api/face/list")
def list_faces():
    result = {}
    for fname in sorted(os.listdir(FACE_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        key = fname.split("_")[0]
        result.setdefault(key, []).append(fname)
    return jsonify(result)


# ------------------------------
# API: 얼굴 DB 캐시 새로고침
# ------------------------------
@app.route("/api/face/reload", methods=["POST"])
def reload_faces():
    global _face_db_cache
    _face_db_cache = rebuild_face_db()
    return jsonify({"ok": True, "count": len(_face_db_cache)})


# ------------------------------
# API: 얼굴 감지
# ------------------------------
@app.route("/api/detect/face", methods=["POST"])
def detect_face():
    file = request.files["file"]
    img = cv_imread_from_upload(file)
    faces = detect_faces(img, _face_cascade)
    return jsonify({"faces": [{"x": x, "y": y, "w": w, "h": h} for (x, y, w, h) in faces]})


# ------------------------------
# API: 얼굴 모자이크 (등록자는 제외)
# ------------------------------
@app.route("/api/mosaic/face", methods=["POST"])
def mosaic_face():
    factor = int(request.args.get("factor", 20))
    file = request.files["file"]
    img = cv_imread_from_upload(file)
    faces = detect_faces(img, _face_cascade)

    ensure_model()
    db = get_face_db()

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # Embedding 계산
        try:
            rep = DeepFace.represent(img_path=face_roi, model_name=MODEL_NAME, enforce_detection=False)
            emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
        except Exception:
            img = apply_mosaic(img, x, y, w, h, factor)
            continue

        # 등록된 얼굴과 비교
        best_name, best_dist = None, 999
        for (name, ref_emb) in db:
            dist = cosine_distance(emb, ref_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        # 매칭 결과 판단
        if best_dist <= COSINE_THRESHOLD:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(img, f"{best_name} ({best_dist:.2f})", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            img = apply_mosaic(img, x, y, w, h, factor)

    _, buf = cv2.imencode(".png", img)
    return send_file(io.BytesIO(buf), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
