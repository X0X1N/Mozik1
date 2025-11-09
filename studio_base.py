# studio_base.py
import os, cv2, time, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageTk
import numpy as np

# 런타임/성능 세팅
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
cv2.setNumThreads(0)

# 프로젝트 내부 유틸
from .core import clamp, gaussian_blur_region, expand_box
from .detect import load_haar_face, detect_faces

# -------- 확장자 --------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS

def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS

# -------- 검출기 --------
try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception:
    _YOLO_OK = False

class YOLOv8FaceDetector:
    def __init__(self, model_path: str = "yolov8n-face.pt", conf: float = 0.45, iou: float = 0.5):
        self._mode = None
        self._yolo = None
        self._haar = None
        self.conf = float(conf)
        self.iou = float(iou)
        self._init_backend(model_path)

    def _init_backend(self, model_path: str):
        if _YOLO_OK:
            try:
                self._yolo = YOLO(model_path)
                self._mode = "yolo"
                print("[INFO] Face detector: YOLOv8-face 활성화됨")
                return
            except Exception as e:
                print(f"[WARN] YOLOv8-face 초기화 실패: {e}")
        try:
            self._haar = load_haar_face()
            self._mode = "haar"
            print("[INFO] Face detector: Haar 폴백 사용")
        except Exception as e:
            self._mode = None
            print(f"[ERROR] Haar 초기화 실패: {e}")

    def detect(self, frame_bgr) -> List[Tuple[int, int, int, int]]:
        if self._mode == "yolo" and self._yolo is not None:
            try:
                results = self._yolo.predict(frame_bgr, imgsz=None, conf=self.conf, iou=self.iou, verbose=False)
                rects = []
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and r.boxes.xyxy is not None:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else None
                        for i, (x1, y1, x2, y2) in enumerate(xyxy):
                            if confs is not None and confs[i] < self.conf:
                                continue
                            x1 = int(max(0, x1)); y1 = int(max(0, y1))
                            x2 = int(max(x1, x2)); y2 = int(max(y1, y2))
                            w = x2 - x1; h = y2 - y1
                            if w >= 12 and h >= 12:
                                rects.append((x1, y1, w, h))
                return rects
            except Exception as e:
                print(f"[WARN] YOLO 예측 오류: {e}")

        if self._mode == "haar" and self._haar is not None:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                return detect_faces(gray, self._haar, 1.1, 5, 24)
            except Exception as e:
                print(f"[ERROR] Haar 검출 실패: {e}")
                return []
        return []

# -------- 상수/데이터 --------
MAX_PREVIEW_W = 1280
DETECT_THUMB_W = 640
THUMB_SIZE = 84
FACE_PAD = 0.18

CENTER_DIST_FACTOR = 1.2
SIZE_SIM_TOL = 0.55

@dataclass
class UBox:
    x: int; y: int; w: int; h: int

@dataclass
class DBox(UBox):
    enabled: bool = True

# -------- 리더 --------
class ImageReader:
    def __init__(self, path: str):
        self.path = os.path.normpath(path)
        img = self._imread_unicode(self.path)
        if img is None:
            try:
                pil = Image.open(self.path).convert("RGB")
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                img = None
        if img is None or img.size == 0:
            raise RuntimeError(f"이미지를 열 수 없음: {self.path}")
        self.img = img

    def _imread_unicode(self, p: str):
        try:
            data = np.fromfile(p, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def read_at(self, index: int) -> Optional[object]:
        return self.img.copy() if index == 0 else None

    def close(self): pass

class VideoReader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없음: {path}")

    def read_at(self, index: int) -> Optional[object]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self.cap.read()
        return frame if ok and frame is not None else None

    def close(self):
        if self.cap is not None:
            self.cap.release()

# -------- 공통 Studio Base --------
class BaseStudio(tk.Tk):
    """
    공통 UI/기능을 모두 포함한 베이스 클래스임이다.
    서브클래스에서 open_media(), export_media(), _start_auto_scan()만 다르게 구현함이다.
    """
    def __init__(self, app_title="Mosaic Studio"):
        super().__init__()
        self.title(app_title)
        self.geometry("1280x820")
        self.style = ttk.Style(self)
        try: self.style.theme_use("clam")
        except: pass

        # 상태
        self.reader: Optional[object] = None
        self.video_path: Optional[str] = None
        self.is_video: bool = False

        self.W = 0; self.H = 0; self.fps = 30.0; self.frame_count = 0
        self.current_idx = 0
        self._scale_preview = 1.0
        self._playing = False
        self._updating_scale = False

        # 박스/락
        self.auto_boxes: Dict[int, List[DBox]] = {}
        self.manual_boxes: List[UBox] = []
        self._auto_lock = threading.Lock()

        # 검출기
        self._detector = YOLOv8FaceDetector(model_path="yolov8n-face.pt", conf=0.45, iou=0.5)

        # 파라미터/UI 상태
        self.strength = tk.IntVar(value=20)
        self.auto_face_enable = tk.BooleanVar(value=True)
        self._thumb_vars: List[tk.BooleanVar] = []
        self._thumb_images: List[ImageTk.PhotoImage] = []

        # 편집 상태
        self._editing_kind: Optional[str] = None
        self._editing_index: Optional[int] = None
        self._resize_anchor: Optional[int] = None
        self._drag_offset: Tuple[int, int] = (0, 0)

        # 스캔 취소
        self._scan_cancel = threading.Event()

        # 전역 제외(해제) 트랙
        self.exclude_tracks: List[Dict] = []
        self._next_track_id = 1

        # UI 구축/바인드
        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- UI ----
    def _build_ui(self):
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="저장 (Ctrl+S)", command=self.export_media)
        menubar.add_cascade(label="파일", menu=filemenu)
        self.config(menu=menubar)

        left_container = ttk.Frame(root); left_container.pack(side="left", fill="y", padx=10, pady=10)
        left_canvas = tk.Canvas(left_container, highlightthickness=0, borderwidth=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side="left", fill="y", expand=False)
        left_scrollbar.pack(side="right", fill="y")
        left = ttk.Frame(left_canvas)
        left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _update_scroll(_=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left.bind("<Configure>", _update_scroll)

        def _on_wheel(event):
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 if event.delta > 0 else 1
            elif getattr(event, "num", None) == 4: delta = -1
            elif getattr(event, "num", None) == 5: delta = 1
            if delta: left_canvas.yview_scroll(delta, "units")
        left_canvas.bind_all("<MouseWheel>", _on_wheel)
        left_canvas.bind_all("<Button-4>", _on_wheel)
        left_canvas.bind_all("<Button-5>", _on_wheel)

        right = ttk.Frame(root); right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ttk.Button(left, text="① 파일 열기", command=self.open_media).pack(fill="x", pady=4)
        sec = ttk.LabelFrame(left, text="② 블러 강도"); sec.pack(fill="x", pady=6)
        ttk.Scale(sec, from_=1, to=60, variable=self.strength, orient="horizontal").pack(fill="x")
        sec2 = ttk.LabelFrame(left, text="③ 자동 얼굴 모자이크"); sec2.pack(fill="x", pady=6)
        ttk.Checkbutton(sec2, text="전체 프레임 자동 탐지", variable=self.auto_face_enable).pack(anchor="w")

        self.face_panel = ttk.LabelFrame(left, text="④ 탐지된 얼굴 (현재 프레임)")
        self.face_panel.pack(fill="x", pady=8)
        top = ttk.Frame(self.face_panel); top.pack(fill="x", pady=(6,4))
        ttk.Button(top, text="현재 프레임 모두 선택", command=lambda: self._toggle_all_detected(True)).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(top, text="현재 프레임 모두 해제", command=lambda: self._toggle_all_detected(False)).pack(side="left", expand=True, fill="x")
        self.thumbs_frame = ttk.Frame(self.face_panel); self.thumbs_frame.pack(fill="x", expand=True, pady=(6,8))

        sec3 = ttk.LabelFrame(left, text="⑤ 추가 블러 프레임 (수동)")
        sec3.pack(fill="x", pady=6)
        rowb = ttk.Frame(sec3); rowb.pack(fill="x")
        ttk.Button(rowb, text="프레임 추가", command=self._add_center_box).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(rowb, text="마지막 삭제", command=self._delete_last_box).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(rowb, text="전체 삭제", command=self._clear_boxes).pack(side="left", expand=True, fill="x")

        sec4 = ttk.LabelFrame(left, text="⑥ 모자이크 제외 대상(영상 전체)")
        sec4.pack(fill="both", expand=False, pady=8)
        self.exclude_list = tk.Listbox(sec4, height=6); self.exclude_list.pack(fill="x", padx=6, pady=6)
        btnrow = ttk.Frame(sec4); btnrow.pack(fill="x", padx=6, pady=(0,6))
        ttk.Button(btnrow, text="선택 삭제", command=self._remove_selected_exclude).pack(side="left", expand=True, fill="x", padx=(0,4))
        ttk.Button(btnrow, text="전체 삭제", command=self._clear_excludes).pack(side="left", expand=True, fill="x")

        self.canvas = tk.Canvas(right, bg="#101010", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        tl = ttk.Frame(right); tl.pack(fill="x", pady=8)
        ttk.Button(tl, text="⏮ 처음", command=lambda: self.seek(0)).pack(side="left")
        ttk.Button(tl, text="▶ 재생", command=self.play).pack(side="left")
        ttk.Button(tl, text="⏸ 정지", command=self.stop).pack(side="left")
        self.scale = ttk.Scale(tl, from_=0, to=0, orient="horizontal", command=self._on_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=8)

        export_frame = ttk.Frame(right); export_frame.pack(fill="x", pady=12)
        self._export_btn = ttk.Button(export_frame, text="⑦ 내보내기", command=self.export_media)
        self._export_btn.pack(fill="x")

    def _bind_shortcuts(self):
        self.bind("<space>", lambda e: (self.stop() if self._playing else self.play()))
        self.bind("<Left>",  lambda e: self.seek(self.current_idx - 1))
        self.bind("<Right>", lambda e: self.seek(self.current_idx + 1))
        self.bind_all("<Control-s>", lambda e: self.export_media())

    # ---- 추상: 서브클래스에서 구현 ----
    def open_media(self):
        raise NotImplementedError

    def export_media(self):
        raise NotImplementedError

    def _start_auto_scan(self):
        raise NotImplementedError

    # ---- 공통 스캔/검출 ----
    def _detect_on_frame(self, frame):
        h, w = frame.shape[:2]
        if w > DETECT_THUMB_W:
            scale = DETECT_THUMB_W / float(w)
            small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        else:
            scale = 1.0
            small = frame
        try:
            rects_small = self._detector.detect(small)
        except Exception:
            rects_small = []
        pad = float(FACE_PAD)
        faces: List[DBox] = []
        for (x, y, w0, h0) in rects_small:
            if scale != 1.0:
                x = int(x / scale); y = int(y / scale)
                w0 = int(w0 / scale); h0 = int(h0 / scale)
            ex, ey, ew, eh = expand_box(x, y, w0, h0, pad, pad, w, h)
            if ew >= 12 and eh >= 12:
                faces.append(DBox(ex, ey, ew, eh, True))
        return faces

    # ---- 타임라인/재생 ----
    def seek(self, idx: int):
        if not self.reader: return
        idx = clamp(idx, 0, self.frame_count - 1)

        frame = None; tries = 0; cur = idx
        while tries < 2 and frame is None:
            frame = self.reader.read_at(cur)
            if frame is None and self.is_video:
                cur = clamp(cur + 1, 0, self.frame_count - 1)
            tries += 1
        if frame is None:
            self.stop(); return

        self.current_idx = cur
        self._build_face_panel(cur, frame)
        self._render_preview(frame)

        self._updating_scale = True
        try: self.scale.set(cur)
        finally: self._updating_scale = False

    def _on_scale(self, v):
        if not self.reader or self._updating_scale: return
        try: self.seek(int(float(v)))
        except: pass

    def play(self):
        if not self.is_video: return
        self._playing = True
        self._loop_play()

    def stop(self):
        self._playing = False

    def _loop_play(self):
        if not self._playing: return
        if self.current_idx >= self.frame_count - 1:
            self.stop(); return
        self.seek(self.current_idx + 1)
        self.after(int(1000 / max(self.fps, 1e-3)), self._loop_play)

    # ---- 전역 제외(해제) ----
    @staticmethod
    def _center(box):
        x, y, w, h = box
        return (x + w * 0.5, y + h * 0.5), max(w, h)

    def _match_track_to_faces(self, track, faces: List[DBox]) -> Optional[int]:
        if not faces: return None
        t_box = track["last_box"]
        (tcx, tcy), tmax = self._center(t_box)
        best_idx, best_score = None, 1e9
        for i, f in enumerate(faces):
            (fcx, fcy), fmax = self._center((f.x, f.y, f.w, f.h))
            dist = ((fcx - tcx) ** 2 + (fcy - tcy) ** 2) ** 0.5
            size_ratio = min(tmax, fmax) / max(tmax, fmax) if max(tmax, fmax) > 0 else 0
            if dist <= max(tmax, fmax) * CENTER_DIST_FACTOR and size_ratio >= SIZE_SIM_TOL:
                if dist < best_score:
                    best_score = dist; best_idx = i
        return best_idx

    def _apply_global_excludes(self, faces: List[DBox]):
        if not self.exclude_tracks or not faces: return
        for track in self.exclude_tracks:
            idx = self._match_track_to_faces(track, faces)
            if idx is None: continue
            faces[idx].enabled = False
            b = faces[idx]
            track["last_box"] = (int(b.x), int(b.y), int(b.w), int(b.h))
            track["last_frame"] = self.current_idx

    # ---- 썸네일 ----
    def _build_face_panel(self, frame_idx: int, frame_bgr):
        with self._auto_lock:
            faces = list(self.auto_boxes.get(frame_idx, []))

        for w in self.thumbs_frame.winfo_children():
            w.destroy()
        self._thumb_vars.clear()
        self._thumb_images.clear()

        if not faces:
            ttk.Label(self.thumbs_frame, text="탐지된 얼굴이 없습니다.").pack(anchor="w", padx=4, pady=4)
            return

        h, w = frame_bgr.shape[:2]
        row = col = 0
        for i, b in enumerate(faces):
            crop = frame_bgr[max(0, b.y):min(h, b.y + b.h), max(0, b.x):min(w, b.x + b.w)]
            if crop.size == 0: continue
            try:
                thumb = cv2.resize(crop, (THUMB_SIZE, THUMB_SIZE))
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
            except Exception:
                continue

            var = tk.BooleanVar(value=b.enabled)
            var.trace_add("write", self._make_thumb_toggle_cb(frame_idx, i, var))

            cell = ttk.Frame(self.thumbs_frame)
            cell.grid(row=row, column=col, padx=4, pady=4, sticky="w")
            ttk.Checkbutton(cell, text=f"Face {i+1} (현재 프레임)", variable=var).pack(anchor="w")
            lbl = ttk.Label(cell, image=imgtk); lbl.image = imgtk; lbl.pack(pady=(2,2))
            ttk.Button(cell, text="이 얼굴 전체 제외", command=lambda i=i, fidx=frame_idx: self._exclude_face_globally(fidx, i)).pack(fill="x")

            self._thumb_vars.append(var); self._thumb_images.append(imgtk)
            col += 1
            if col >= 3: col = 0; row += 1

    def _make_thumb_toggle_cb(self, frame_idx: int, i: int, var: tk.BooleanVar):
        def _cb(*_):
            with self._auto_lock:
                lst = self.auto_boxes.get(frame_idx, [])
                if 0 <= i < len(lst):
                    lst[i].enabled = bool(var.get())
            self.seek(self.current_idx)
        return _cb

    def _toggle_all_detected(self, flag: bool):
        with self._auto_lock:
            lst = self.auto_boxes.get(self.current_idx, [])
            for b in lst: b.enabled = flag
        self.seek(self.current_idx)

    # ---- 프리뷰 렌더 ----
    def _fit_to_canvas(self, frame):
        cw = self.canvas.winfo_width() or MAX_PREVIEW_W
        ch = self.canvas.winfo_height() or int(MAX_PREVIEW_W * 9 / 16)
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame, 1.0, 0, 0
        scale = max(1e-6, min(cw / w, ch / h))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        ox = (cw - nw) // 2; oy = (ch - nh) // 2
        return cv2.resize(frame, (nw, nh)), scale, ox, oy

    def _render_preview(self, frame_bgr):
        vis = frame_bgr.copy()
        with self._auto_lock:
            auto_list = list(self.auto_boxes.get(self.current_idx, []))
        self._apply_global_excludes(auto_list)

        for b in auto_list:
            if b.enabled:
                gaussian_blur_region(vis, b.x, b.y, b.w, b.h, int(self.strength.get()))
        for b in self.manual_boxes:
            gaussian_blur_region(vis, b.x, b.y, b.w, b.h, int(self.strength.get()))

        img, s, ox, oy = self._fit_to_canvas(vis)
        self._scale_preview = s
        self.canvas.delete("all")
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(pil)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(ox, oy, anchor="nw", image=imgtk)

        # 보라(자동) / 시안(수동)
        with self._auto_lock:
            auto_list_draw = list(self.auto_boxes.get(self.current_idx, []))
        for b in auto_list_draw:
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            if not b.enabled:
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#A56BFF", width=1)
            else:
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#A56BFF", width=2)
                r = 5
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(hx - r, hy - r, hx + r, hy + r, fill="#A56BFF", outline="")
        for b in self.manual_boxes:
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#5EC2FF", width=2)

    # ---- 좌표/히트 ----
    def _canvas_to_video(self, cx, cy):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        s = max(self._scale_preview, 1e-6)
        nw, nh = int(self.W * s), int(self.H * s)
        ox = (cw - nw) // 2; oy = (ch - nh) // 2
        x = int((cx - ox) / s); y = int((cy - oy) / s)
        return clamp(x, 0, self.W - 1), clamp(y, 0, self.H - 1)

    def _canvas_offsets(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        s = max(self._scale_preview, 1e-6)
        nw, nh = int(self.W * s), int(self.H * s)
        ox = (cw - nw) // 2; oy = (ch - nh) // 2
        return s, ox, oy

    def _hit_handle_generic(self, cx, cy):
        s, ox, oy = self._canvas_offsets()
        with self._auto_lock:
            auto_list = list(self.auto_boxes.get(self.current_idx, []))
        for i, b in enumerate(auto_list):
            if not b.enabled: continue
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            for idx, (hx, hy) in enumerate([(x1, y1), (x2, y1), (x1, y2), (x2, y2)]):
                if abs(cx - hx) <= 6 and abs(cy - hy) <= 6:
                    return "auto", i, idx
        for i, b in enumerate(self.manual_boxes):
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            for idx, (hx, hy) in enumerate([(x1, y1), (x2, y1), (x1, y2), (x2, y2)]):
                if abs(cx - hx) <= 6 and abs(cy - hy) <= 6:
                    return "manual", i, idx
        return None, None, None

    def _hit_box_generic(self, cx, cy):
        s, ox, oy = self._canvas_offsets()
        with self._auto_lock:
            auto_list = list(self.auto_boxes.get(self.current_idx, []))
        for i, b in enumerate(auto_list):
            if not b.enabled: continue
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            if x1 <= cx <= x2 and y1 <= cy <= y2: return "auto", i
        for i, b in enumerate(self.manual_boxes):
            x1 = int(b.x * s) + ox; y1 = int(b.y * s) + oy
            x2 = int((b.x + b.w) * s) + ox; y2 = int((b.y + b.h) * s) + oy
            if x1 <= cx <= x2 and y1 <= cy <= y2: return "manual", i
        return None, None

    # ---- 마우스 ----
    def _on_mouse_down(self, e):
        kind, idx, corner = self._hit_handle_generic(e.x, e.y)
        if kind is not None:
            self._editing_kind = kind; self._editing_index = idx; self._resize_anchor = corner; return
        kind, idx = self._hit_box_generic(e.x, e.y)
        if kind is not None:
            self._editing_kind = kind; self._editing_index = idx; self._resize_anchor = None
            vx, vy = self._canvas_to_video(e.x, e.y)
            if kind == "manual":
                b = self.manual_boxes[idx]
            else:
                with self._auto_lock: b = self.auto_boxes.get(self.current_idx, [])[idx]
            self._drag_offset = (vx - b.x, vy - b.y); return
        self._editing_kind = self._editing_index = self._resize_anchor = None
        self._new_box_start = (e.x, e.y)

    def _on_mouse_drag(self, e):
        if self._editing_kind is None or self._editing_index is None: return
        vx, vy = self._canvas_to_video(e.x, e.y)
        if self._editing_kind == "manual":
            b = self.manual_boxes[self._editing_index]
        else:
            with self._auto_lock:
                lst = self.auto_boxes.get(self.current_idx, [])
                if not (0 <= self._editing_index < len(lst)): return
                b = lst[self._editing_index]

        if self._resize_anchor is not None:
            x2, y2 = b.x + b.w, b.y + b.h
            if self._resize_anchor == 0:
                nx, ny = vx, vy; nw, nh = x2 - nx, y2 - ny
                if nw > 8 and nh > 8: b.x, b.y, b.w, b.h = nx, ny, nw, nh
            elif self._resize_anchor == 1:
                nx, ny = b.x, vy; nw, nh = vx - b.x, y2 - ny
                if nw > 8 and nh > 8: b.y, b.w, b.h = ny, nw, nh
            elif self._resize_anchor == 2:
                nx, ny = vx, b.y; nw, nh = x2 - nx, vy - b.y
                if nw > 8 and nh > 8: b.x, b.w, b.h = nx, nw, nh
            else:
                nw, nh = vx - b.x, vy - b.y
                if nw > 8 and nh > 8: b.w, b.h = nw, nh
        else:
            nx = clamp(vx - self._drag_offset[0], 0, self.W - b.w)
            ny = clamp(vy - self._drag_offset[1], 0, self.H - b.h)
            b.x, b.y = nx, ny
        self.seek(self.current_idx)

    def _on_mouse_up(self, e):
        if hasattr(self, "_new_box_start") and self._new_box_start:
            sx, sy = self._new_box_start
            vx1, vy1 = self._canvas_to_video(sx, sy)
            vx2, vy2 = self._canvas_to_video(e.x, e.y)
            x = min(vx1, vx2); y = min(vy1, vy2)
            w = abs(vx2 - vx1); h = abs(vy2 - vy1)
            if w > 8 and h > 8:
                self.manual_boxes.append(UBox(x, y, w, h))
                self.seek(self.current_idx)
        self._editing_kind = self._editing_index = self._resize_anchor = None
        self._new_box_start = None

    # ---- 수동 박스 유틸 ----
    def _add_center_box(self):
        if self.W <= 0 or self.H <= 0: return
        w, h = max(32, int(self.W * 0.3)), max(32, int(self.H * 0.3))
        x, y = (self.W - w) // 2, (self.H - h) // 2
        self.manual_boxes.append(UBox(x, y, w, h))
        self.seek(self.current_idx)

    def _delete_last_box(self):
        if self.manual_boxes:
            self.manual_boxes.pop()
            self.seek(self.current_idx)

    def _clear_boxes(self):
        self.manual_boxes.clear()
        self.seek(self.current_idx)

    # ---- 제외 등록/삭제 ----
    def _exclude_face_globally(self, frame_idx: int, i: int):
        with self._auto_lock:
            lst = self.auto_boxes.get(frame_idx, [])
            if not (0 <= i < len(lst)): return
            b = lst[i]
            track = {
                "id": self._next_track_id,
                "init_frame": frame_idx,
                "last_frame": frame_idx,
                "last_box": (int(b.x), int(b.y), int(b.w), int(b.h))
            }
            self.exclude_tracks.append(track)
            self.exclude_list.insert(tk.END, f"ID {track['id']} @F{frame_idx} ({b.w}x{b.h})")
            self._next_track_id += 1
        self.seek(self.current_idx)

    def _remove_selected_exclude(self):
        sel = list(self.exclude_list.curselection())
        if not sel: return
        sel.sort(reverse=True)
        for idx in sel:
            self.exclude_list.delete(idx)
            if 0 <= idx < len(self.exclude_tracks):
                self.exclude_tracks.pop(idx)
        self.seek(self.current_idx)

    def _clear_excludes(self):
        self.exclude_tracks.clear()
        self.exclude_list.delete(0, tk.END)
        self.seek(self.current_idx)

    # ---- 종료 ----
    def _on_close(self):
        self._scan_cancel.set()
        self.stop()
        try:
            if self.reader: self.reader.close()
        except: pass
        self.destroy()
