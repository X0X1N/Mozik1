# create_authorized_face.py
import os
import cv2
import time

# -----------------------------------
# 기본 설정
# -----------------------------------
SAVE_DIR = "detected_face"   # 여러 사람의 얼굴 이미지를 저장하는 폴더

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# -----------------------------------
# 유틸 함수
# -----------------------------------
def try_open_camera(max_index: int = 3):
    """
    여러 인덱스와 여러 백엔드로 카메라 열기를 시도한다.
    우선순위: DirectShow -> MSMF -> 기본
    성공 시 (cap, index, backend_name) 반환, 실패 시 (None, None, None) 반환이다.
    """
    backends = [
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_MSMF,  "CAP_MSMF"),
        (None,          "DEFAULT"),
    ]
    for idx in range(0, max_index + 1):
        for backend, bname in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
                if cap is not None and cap.isOpened():
                    # 해상도 기본값 설정(필요 시 조절 가능)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print(f"[INFO] Camera opened at index={idx}, backend={bname}")
                    return cap, idx, bname
                if cap is not None:
                    cap.release()
            except Exception as e:
                if cap is not None:
                    cap.release()
                print(f"[WARN] open failed: index={idx}, backend={bname}, err={e}")
    return None, None, None

def pick_largest_face(faces):
    """
    여러 얼굴 중 가장 큰(면적 최대) 얼굴 1개를 고른다.
    faces는 (x, y, w, h) 튜플 리스트이다.
    """
    if len(faces) == 0:
        return None
    areas = [(w * h, (x, y, w, h)) for (x, y, w, h) in faces]
    areas.sort(key=lambda t: t[0], reverse=True)
    return areas[0][1]

def padded_crop(frame, box, pad_ratio=0.25):
    """
    얼굴 박스(box)에 pad_ratio 만큼 여유 패딩을 주어 잘리지 않게 크롭한다.
    """
    (x, y, w, h) = box
    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    img_h, img_w = frame.shape[:2]
    new_x = max(0, x - pad_w)
    new_y = max(0, y - pad_h)
    new_w = min(img_w - new_x, w + 2 * pad_w)
    new_h = min(img_h - new_y, h + 2 * pad_h)

    return frame[new_y:new_y+new_h, new_x:new_x+new_w], (new_x, new_y, new_w, new_h)

# -----------------------------------
# 메인 로직
# -----------------------------------
def capture_and_save_faces_multi():
    print("[INFO] 얼굴 등록을 시작한다. 's'=저장, 'n'=이름변경, 'q'=종료이다.")

    # 카메라 열기(여러 인덱스/백엔드 폴백)
    cap, cam_idx, backend_name = try_open_camera()
    if cap is None:
        print("[ERROR] 웹캠을 열 수 없다. 다른 앱 점유/권한/드라이버 상태를 확인하라.")
        return

    # Haar 로드
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_path):
        print(f"[ERROR] haarcascade 파일을 찾을 수 없다: {haar_path}")
        cap.release()
        return

    face_cascade = cv2.CascadeClassifier(haar_path)
    if face_cascade.empty():
        print("[ERROR] Haar Cascade 로드를 실패했다.")
        cap.release()
        return

    # 최초 등록자 이름 입력
    person_name = input("[INPUT] 현재 등록할 이름을 입력하라(예: honggildong): ").strip()
    if not person_name:
        print("[ERROR] 이름이 비어있다. 종료한다.")
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임을 읽지 못했다. 종료한다.")
            break

        frame = cv2.flip(frame, 1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 탐지
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        largest = pick_largest_face(faces)

        # 안내 텍스트
        header = f"Cam:{cam_idx}/{backend_name} | Name:{person_name} | 's':save  'n':name  'q':quit"
        cv2.putText(frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        if largest is not None:
            (x, y, w, h) = largest
            # 박스 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(frame, "Press 's' to save", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

        cv2.imshow("Register Faces (multi-user) - s:save / n:name / q:quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 종료
            break

        elif key == ord('n'):
            # 이름 변경
            new_name = input("[INPUT] 새로 등록할 이름: ").strip()
            if new_name:
                person_name = new_name
                print(f"[INFO] 현재 등록 대상 이름을 '{person_name}'(으)로 변경했다.")

        elif key == ord('s'):
            # 저장
            if largest is None:
                print("[WARNING] 저장할 얼굴이 탐지되지 않았다. 다시 시도하라.")
                continue

       
            cropped, padded_box = padded_crop(frame, largest, pad_ratio=0.25)

    
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"{person_name}_{ts}.jpg")
            cv2.imwrite(save_path, cropped)
            print(f"[SUCCESS] '{person_name}' 얼굴을 '{save_path}'에 저장했다.")

            

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 등록을 종료한다.")


if __name__ == '__main__':
    capture_and_save_faces_multi()
