# studio_video.py
import os, cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from .studio_base import BaseStudio, VIDEO_EXTS, is_video, VideoReader, gaussian_blur_region

class VideoStudio(BaseStudio):
    def __init__(self):
        super().__init__(app_title="Mosaic Studio — Video")

    def open_media(self):
        patterns_vid = ";".join(f"*{ext}" for ext in sorted(VIDEO_EXTS))
        path = filedialog.askopenfilename(
            title="동영상 선택",
            filetypes=[("동영상", patterns_vid), ("모든 파일", "*.*")]
        )
        if not path: return
        path = os.path.normpath(path)

        if not is_video(path):
            messagebox.showerror("오류", "동영상 파일이 아님이다.")
            return

        # 이전 상태 정리
        self._scan_cancel.set()
        if self.reader: self.reader.close()

        # 비디오 로드
        self.reader = VideoReader(path)
        self.video_path = path
        cap = self.reader.cap
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.is_video = True

        self._export_btn.config(text="⑦ 동영상 내보내기 (Ctrl+S)")

        # 상태 초기화
        self.auto_boxes.clear(); self.manual_boxes.clear()
        self.exclude_tracks.clear(); self.exclude_list.delete(0, "end")
        self.scale.configure(to=max(self.frame_count - 1, 0))

        self.seek(0)
        self._start_auto_scan()

    def _start_auto_scan(self):
        if not self.auto_face_enable.get() or not self.video_path: return

        def _scan_all():
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if not cap or not cap.isOpened(): return
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.frame_count
                for i in range(total):
                    if self._scan_cancel.is_set(): break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    if not ok or frame is None: continue
                    faces = self._detect_on_frame(frame)
                    with self._auto_lock:
                        self.auto_boxes[i] = faces
                    if i == self.current_idx and not self._scan_cancel.is_set():
                        self.after(0, lambda idx=i, f=frame.copy(): (self._build_face_panel(idx, f), self.seek(idx)))
            finally:
                try: cap.release()
                except: pass
            if not self._scan_cancel.is_set():
                self.after(0, lambda: self.seek(self.current_idx))

        self._scan_cancel.clear()
        import threading; threading.Thread(target=_scan_all, daemon=True).start()

    def export_media(self):
        if not self.reader:
            messagebox.showinfo("안내", "먼저 동영상을 여세요.")
            return

        out = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if not out: return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, self.fps, (self.W, self.H))
        if not writer.isOpened():
            messagebox.showerror("오류", "출력 파일을 열 수 없습니다.")
            return

        top = tk.Toplevel(self); top.title("내보내는 중…")
        pb = ttk.Progressbar(top, maximum=self.frame_count, length=360); pb.pack(padx=20, pady=16)
        msg = ttk.Label(top, text="0%"); msg.pack()

        try:
            for i in range(self.frame_count):
                frame = self.reader.read_at(i)
                if frame is None: continue

                with self._auto_lock:
                    auto_list = list(self.auto_boxes.get(i, []))
                self.current_idx = i
                self._apply_global_excludes(auto_list)

                for b in auto_list:
                    if b.enabled:
                        gaussian_blur_region(frame, b.x, b.y, b.w, b.h, int(self.strength.get()))
                for b in self.manual_boxes:
                    gaussian_blur_region(frame, b.x, b.y, b.w, b.h, int(self.strength.get()))
                writer.write(frame)

                if i % 5 == 0:
                    pb['value'] = i + 1
                    msg.config(text=f"{100.0 * (i + 1) / max(self.frame_count,1):.1f}%")
                    self.update()
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror("오류", f"내보내기 중 오류: {e}")
        finally:
            try: writer.release()
            except: pass
            try: top.destroy()
            except: pass
        messagebox.showinfo("완료", f"저장됨: {out}")

def run():
    app = VideoStudio()
    app.mainloop()

if __name__ == "__main__":
    run()
