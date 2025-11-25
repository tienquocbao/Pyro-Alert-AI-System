import cv2
import numpy as np
import os
import math
import time
import datetime
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from collections import deque
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# ================= CẤU HÌNH HỆ THỐNG & THÔNG BÁO =================

# 1. Cấu hình Model & Video
MODEL_PATH = "models/Yolov8s-50epochs.pt"
VIDEO_PATHS = ["videos/firevid2.mp4", "videos/nonfirevid1.mp4", "videos/nonfirevid2.mp4", "videos/nonfirevid3.mp4"]
OUTPUT_DIR = "runs/alerts"
CONF_THRESHOLD = 0.45
CLASS_ID_FIRE = 0

# 2. Cấu hình Telegram (Bạn cần điền token của bạn)
TELEGRAM_ENABLE = True
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN" 
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

# 3. Cấu hình Email (Dùng App Password nếu là Gmail)
EMAIL_ENABLE = True
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Không dùng mật khẩu đăng nhập, dùng App Password
EMAIL_RECEIVER = "admin_email@gmail.com"

# 4. Cấu hình Cooldown (Chống Spam)
NOTIFICATION_COOLDOWN = 60 # Chỉ gửi cảnh báo mỗi 60 giây cho 1 camera

# Cấu hình Grid View
RESIZE_W = 640
RESIZE_H = 360
GAP_SIZE = 4
GAP_COLOR = (20, 20, 20)

# Cấu hình quay video
FPS_RECORD = 30
PRE_PADDING = 2 * FPS_RECORD
POST_PADDING = 3 * FPS_RECORD

os.makedirs(OUTPUT_DIR, exist_ok=True)
app = Flask(__name__)
model = YOLO(MODEL_PATH, task='detect')

# ================= MODULE THÔNG BÁO (NOTIFICATION SYSTEM) =================
class NotificationSystem:
    @staticmethod
    def send_telegram(cam_id, image_frame):
        if not TELEGRAM_ENABLE: return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            # Encode ảnh để gửi
            _, img_encoded = cv2.imencode('.jpg', image_frame)
            files = {'photo': ('fire.jpg', img_encoded.tobytes())}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': f"🔥 CẢNH BÁO: Phát hiện LỬA tại {cam_id}!"}
            requests.post(url, files=files, data=data)
            print(f"[TELEGRAM] Đã gửi cảnh báo cho {cam_id}")
        except Exception as e:
            print(f"[LỖI TELEGRAM] {e}")

    @staticmethod
    def send_email(cam_id, image_frame, time_str):
        if not EMAIL_ENABLE: return
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"🔥 FIRE ALERT: {cam_id} - {time_str}"
            msg['From'] = EMAIL_SENDER
            msg['To'] = EMAIL_RECEIVER

            text = MIMEText(f"Hệ thống phát hiện nguy cơ cháy tại {cam_id} vào lúc {time_str}.\nVui lòng kiểm tra ngay.")
            msg.attach(text)

            # Đính kèm ảnh
            _, img_encoded = cv2.imencode('.jpg', image_frame)
            image = MIMEImage(img_encoded.tobytes(), name=f"fire_{cam_id}.jpg")
            msg.attach(image)

            # Gửi mail qua SMTP Gmail
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            print(f"[EMAIL] Đã gửi mail cảnh báo cho {cam_id}")
        except Exception as e:
            print(f"[LỖI EMAIL] {e}")

    @staticmethod
    def trigger_alerts(cam_id, frame):
        # Chạy trong luồng riêng để không làm lag video
        time_str = datetime.datetime.now().strftime("%H:%M:%S %d/%m/%Y")
        
        t1 = threading.Thread(target=NotificationSystem.send_telegram, args=(cam_id, frame))
        t2 = threading.Thread(target=NotificationSystem.send_email, args=(cam_id, frame, time_str))
        
        t1.start()
        t2.start()

# ================= CLASS XỬ LÝ CAMERA (CHỈ RESIZE - KHÔNG SKIP FRAME) =================
class CameraAgent:
    def __init__(self, cam_id, source_path):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(source_path)
        
        # Buffer ghi hình
        self.buffer = deque(maxlen=PRE_PADDING)
        self.is_recording = False
        self.recording_writer = None
        self.post_record_counter = 0
        self.current_filename = ""
        self.last_alert_time = 0 

        # Thông số gốc của video
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Kích thước ảnh đầu vào cho AI (Chuẩn YOLOv8 là 640)
        self.AI_SIZE = 640 

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def process_logic(self, frame):
        # 1. Chuẩn bị Frame hiển thị (Giữ nguyên gốc)
        display_frame = frame.copy()

        # 2. Resize Frame cho AI (Giảm tải cho GPU)
        # YOLOv8 được train trên ảnh vuông 640x640, nên resize về đây là nhanh và chuẩn nhất
        ai_frame = cv2.resize(frame, (self.AI_SIZE, self.AI_SIZE))

        # 3. Chạy Detect (Chạy trên mọi frame, không skip)
        results = model(ai_frame, verbose=False, conf=CONF_THRESHOLD, iou=0.45, device=0)
        
        detected = False
        
        # Tính tỷ lệ để phóng to toạ độ box từ 640x640 về kích thước thật
        scale_x = self.width / self.AI_SIZE
        scale_y = self.height / self.AI_SIZE

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == CLASS_ID_FIRE:
                    detected = True
                    
                    # Lấy toạ độ trên ảnh nhỏ (ai_frame)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Scale toạ độ về ảnh lớn (display_frame)
                    real_x1 = int(x1 * scale_x)
                    real_y1 = int(y1 * scale_y)
                    real_x2 = int(x2 * scale_x)
                    real_y2 = int(y2 * scale_y)

                    # Vẽ box lên ảnh gốc
                    cv2.rectangle(display_frame, (real_x1, real_y1), (real_x2, real_y2), (0, 0, 255), 2)
                    cv2.putText(display_frame, f"FIRE {self.cam_id}", (real_x1, real_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. Logic Thông Báo & Ghi Hình (Giữ nguyên như cũ)
        current_timestamp = time.time()
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if detected:
            if current_timestamp - self.last_alert_time > NOTIFICATION_COOLDOWN:
                # Gửi ảnh display_frame (ảnh nét) đi cảnh báo
                NotificationSystem.trigger_alerts(self.cam_id, display_frame)
                self.last_alert_time = current_timestamp
                GlobalState.should_play_sound = True

        if not self.is_recording:
            self.buffer.append(frame) # Lưu frame gốc vào buffer
            if detected:
                self.is_recording = True
                self.current_filename = f"{self.cam_id}_{current_time_str}.mp4"
                save_path = os.path.join(OUTPUT_DIR, self.current_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.recording_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (self.width, self.height))
                while self.buffer: self.recording_writer.write(self.buffer.popleft())
                self.recording_writer.write(frame)
                GlobalState.alerts.insert(0, {"cam": self.cam_id, "time": datetime.datetime.now().strftime("%H:%M:%S"), "file": self.current_filename})
        else:
            if detected:
                self.post_record_counter = POST_PADDING
                self.recording_writer.write(frame)
            else:
                if self.post_record_counter > 0:
                    self.recording_writer.write(frame)
                    self.post_record_counter -= 1
                else:
                    self.is_recording = False
                    self.recording_writer.release()
                    self.recording_writer = None

        return display_frame
# ================= GLOBAL STATE =================
class GlobalState:
    cameras = []
    alerts = []
    should_play_sound = False # Biến cờ để báo hiệu Web UI phát âm thanh

# Khởi tạo Camera
for i, path in enumerate(VIDEO_PATHS):
    if os.path.exists(path):
        GlobalState.cameras.append(CameraAgent(f"Cam_{i+1}", path))

def create_mosaic_frame(frames):
    n = len(frames)
    if n == 0: return None
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    total_w = RESIZE_W * cols + GAP_SIZE * (cols + 1)
    total_h = RESIZE_H * rows + GAP_SIZE * (rows + 1)
    canvas = np.full((total_h, total_w, 3), GAP_COLOR, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= n: continue
            small_frame = cv2.resize(frames[idx], (RESIZE_W, RESIZE_H))
            x = GAP_SIZE + c * (RESIZE_W + GAP_SIZE)
            y = GAP_SIZE + r * (RESIZE_H + GAP_SIZE)
            canvas[y:y+RESIZE_H, x:x+RESIZE_W] = small_frame
    return canvas

def generate_feed():
    frame_count = 0
    start_time = time.time()
    current_fps = 0.0

    while True:
        frames_to_merge = []
        for cam in GlobalState.cameras:
            ret, raw = cam.read_frame()
            if ret: frames_to_merge.append(cam.process_logic(raw))
            else: frames_to_merge.append(np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8))

        if frames_to_merge:
            mosaic = create_mosaic_frame(frames_to_merge)
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(mosaic, f"FPS: {current_fps:.1f}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            ret, buffer = cv2.imencode('.jpg', mosaic)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ================= FLASK ROUTER =================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/alerts')
def get_alerts():
    return jsonify(GlobalState.alerts[:15])

@app.route('/api/status')
def check_status():
    # API này để Web UI hỏi xem có cần hú còi không
    play = GlobalState.should_play_sound
    if play:
        GlobalState.should_play_sound = False # Reset sau khi đã báo
    return jsonify({"play_sound": play})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)