# main.py
import os
import cv2
import numpy as np
from deepface import DeepFace
import time

FACE_DATABASE_DIR = "detected_face"
MODEL_NAME = "VGG-Face"
DISTANCE_METRIC = "cosine"
MOSAIC_FACTOR = 20
COSINE_THRESHOLD = 0.40   # 등록 환경에 따라 0.35~0.55 범위 튜닝 권장

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# -----------------------------
# 유틸
# -----------------------------
def apply_mosaic(image, face_area):
    x, y, w, h = face_area
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        return image
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0:
        return image
    small = cv2.resize(face_roi, (max(1, w // MOSAIC_FACTOR), max(1, h // MOSAIC_FACTOR)),
                       interpolation=cv2.INTER_NEAREST)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    image[y:y+h, x:x+w] = mosaic
    return image

def cosine_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b) / denom)

def parse_person_name(basename_without_ext: str) -> str:
    """
    파일명에서 사람 이름 부분만 추출한다.
    규칙: 이름_타임스탬프.jpg 형태를 가정하고, 마지막 '_' 앞까지를 이름으로 간주한다.
    예) hong_gil_dong_20251102_210501 -> hong_gil_dong
    """
    if "_" in basename_without_ext:
        name, _ = basename_without_ext.rsplit("_", 1)
        return name
    return basename_without_ext

def build_model_once():
    print("[INFO] 얼굴 인식 모델 로드 중이다...")
    DeepFace.build_model(MODEL_NAME)
    print("[INFO] 모델 로드 완료이다.")

def compute_embedding_from_image(img_bgr):
    """
    BGR 이미지에서 DeepFace 임베딩을 계산한다.
    """
    reps = DeepFace.represent(
        img_path=img_bgr,
        model_name=MODEL_NAME,
        enforce_detection=False
    )
    emb = reps[0]["embedding"] if isinstance(reps, list) else reps["embedding"]
    return np.array(emb, dtype=float)

# -----------------------------
# 등록 얼굴 로드(사람별 평균 임베딩)
# -----------------------------
def load_authorized_embeddings_grouped():
    """
    detected_face 폴더의 모든 이미지에서 사람별 임베딩 평균을 만든다.
    반환: List[ (person_name, mean_embedding) ]
    """
    if not os.path.isdir(FACE_DATABASE_DIR):
        print(f"[ERROR] 폴더가 없다: {FACE_DATABASE_DIR}")
        return []

    # 사람별 임베딩 리스트
    buckets = {}  # name -> list of embeddings

    files = [f for f in os.listdir(FACE_DATABASE_DIR)
             if f.lower().endswith(SUPPORTED_EXTS)]
    if not files:
        print("[ERROR] 등록된 얼굴 이미지가 없다.")
        return []

    for fname in files:
        path = os.path.join(FACE_DATABASE_DIR, fname)
        base = os.path.splitext(fname)[0]
        person = parse_person_name(base)
        try:
            reps = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                enforce_detection=False
            )
            emb = reps[0]["embedding"] if isinstance(reps, list) else reps["embedding"]
            emb = np.array(emb, dtype=float)
            buckets.setdefault(person, []).append(emb)
            print(f"[EMBED] {person} ← {fname} 완료")
        except Exception as e:
            print(f"[WARN] 임베딩 실패: {fname} / {e}")

    entries = []
    for person, embs in buckets.items():
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        entries.append((person, mean_emb))
    print(f"[INFO] 등록 인원 수: {len(entries)}명 "
          f"(총 파일 {len(files)}장, 사람별 평균 임베딩 반영)")
    return entries

# -----------------------------
# 메인
# -----------------------------
def run_mosaic_app():
    # 얼굴 검출기(Haar)
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print(f"[ERROR] Haar Cascade 파일을 찾을 수 없다: {haar_path}")
        return
    face_cascade = cv2.CascadeClassifier(haar_path)

    # 카메라 열기
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없다. 다른 앱 점유/권한 설정을 확인하라.")
        return

    # 모델 로드
    try:
        build_model_once()
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        cap.release()
        return

    # 등록 얼굴 로드
    db_entries = load_authorized_embeddings_grouped()
    if not db_entries:
        print(f"[ERROR] '{FACE_DATABASE_DIR}'에 등록 얼굴이 없다.")
        cap.release()
        return

    print("----------------------------------------------------")
    print("[INFO] 실시간 감지를 시작한다. 'q': 종료, 'r': 등록 얼굴 재로드이다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임을 읽지 못했다. 종료한다.")
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            # 현재 얼굴 임베딩 계산
            try:
                cur_emb = compute_embedding_from_image(face_img)
            except Exception as e:
                print(f"[WARN] 현재 얼굴 임베딩 실패: {e}")
                frame = apply_mosaic(frame, (x, y, w, h))
                continue

            # 사람별 평균 임베딩과 거리 비교
            best_name = None
            best_dist = 1e9
            for person, mean_emb in db_entries:
                dist = cosine_distance(cur_emb, mean_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_name = person

            if best_dist <= COSINE_THRESHOLD:
                # 등록된 얼굴 → 모자이크하지 않음
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 0), 2)
                cv2.putText(frame, f"{best_name} ({best_dist:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            else:
                # 미등록 얼굴 → 모자이크
                frame = apply_mosaic(frame, (x, y, w, h))
                cv2.putText(frame, f"Unknown ({best_dist:.2f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 2)

        cv2.imshow("Real-time Mosaic (Multi-User) - q:quit / r:reload", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] 등록 얼굴을 다시 로드한다.")
            db_entries = load_authorized_embeddings_grouped()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 프로그램을 종료한다.")

if __name__ == '__main__':
    run_mosaic_app()
