# studio_image.py
import os, cv2
from tkinter import filedialog, messagebox
from PIL import Image

# ✅ BaseStudio와 이미지 유틸은 studio_base에서, 블러 함수는 core에서 가져옴
from .studio_base import BaseStudio, IMAGE_EXTS, is_image, ImageReader
from .core import gaussian_blur_region

class ImageStudio(BaseStudio):
    def __init__(self):
        super().__init__(app_title="Mosaic Studio — Image")

    def open_media(self):
        # ✅ 일부 환경에서 세미콜론(;) 대신 공백 구분이 더 호환성이 좋음
        patterns_img = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTS))

        path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("이미지", patterns_img), ("모든 파일", "*.*")]
        )
        if not path:
            return
        path = os.path.normpath(path)

        # ✅ 확장자 세트에 없는 형식을 대비해 JFIF/HEIC 등을 추가적으로 허용할 수도 있음
        if not is_image(path):
            # 추가 허용 확장자(필요 시): .jfif, .heic, .heif
            extra_exts = {".jfif", ".heic", ".heif"}
            if os.path.splitext(path)[1].lower() not in extra_exts:
                messagebox.showerror("오류", "이미지 파일이 아님이다.")
                return

        # 이전 상태 정리
        self._scan_cancel.set()
        if self.reader:
            self.reader.close()

        # 이미지 로드
        try:
            self.reader = ImageReader(path)
            frame0 = self.reader.read_at(0)
            if frame0 is None or frame0.size == 0:
                raise RuntimeError("이미지를 읽을 수 없음")
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
            self.reader = None
            return

        self.W = int(frame0.shape[1]); self.H = int(frame0.shape[0])
        self.fps = 1.0
        self.frame_count = 1
        self.is_video = False

        self._export_btn.config(text="⑦ 이미지 저장 (Ctrl+S)")

        # 상태 초기화
        self.auto_boxes.clear(); self.manual_boxes.clear()
        self.exclude_tracks.clear(); self.exclude_list.delete(0, "end")
        self.scale.configure(to=0)  # 프레임 1장

        # 첫 프레임 표시
        self.seek(0)

        # 자동 스캔
        self._start_auto_scan()

    def _start_auto_scan(self):
        if not self.auto_face_enable.get() or not self.reader:
            return

        # 단일 프레임 스캔
        def _scan():
            if self._scan_cancel.is_set():
                return
            frame = self.reader.read_at(0)
            if frame is None:
                return
            faces = self._detect_on_frame(frame)
            self.auto_boxes[0] = faces
            if not self._scan_cancel.is_set():
                self.after(0, lambda: self.seek(0))

        self._scan_cancel.clear()
        import threading
        threading.Thread(target=_scan, daemon=True).start()

    def export_media(self):
        if not self.reader:
            messagebox.showinfo("안내", "먼저 이미지를 여세요.")
            return

        out = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
                ("WEBP", "*.webp"),
                ("TIFF", "*.tif;*.tiff"),
                ("모든 파일", "*.*"),
            ],
        )
        if not out:
            return

        frame = self.reader.read_at(0)
        if frame is None:
            messagebox.showerror("오류", "이미지를 읽을 수 없습니다.")
            return

        auto_list = list(self.auto_boxes.get(0, []))
        self.current_idx = 0
        self._apply_global_excludes(auto_list)

        for b in auto_list:
            if b.enabled:
                gaussian_blur_region(frame, b.x, b.y, b.w, b.h, int(self.strength.get()))
        for b in self.manual_boxes:
            gaussian_blur_region(frame, b.x, b.y, b.w, b.h, int(self.strength.get()))

        ext = os.path.splitext(out)[1].lower()
        encode_flag = {
            ".png": ".png", ".jpg": ".jpg", ".jpeg": ".jpg",
            ".bmp": ".bmp", ".webp": ".webp", ".tif": ".tif", ".tiff": ".tif",
        }.get(ext, ".png")

        try:
            ok, enc = cv2.imencode(encode_flag, frame)
            if not ok:
                raise RuntimeError("이미지 인코딩 실패")
            enc.tofile(out)
        except Exception:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out)
        messagebox.showinfo("완료", f"저장됨: {out}")

def run():
    app = ImageStudio()
    app.mainloop()

if __name__ == "__main__":
    run()
