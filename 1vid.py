import cv2
from ultralytics import YOLO

# --- CẤU HÌNH ---
VIDEO_PATH = "videos\\firevid2.mp4"       # Đường dẫn video input
MODEL_PATH = "models\\Yolov8s-50epochs.pt"       # Đường dẫn model (hoặc 'models/Yolov8n-p2-50epochs.pt')
CONF_THRESHOLD = 0.5            # Độ tin cậy (0.5 = 50%)

# 1. Load Model
print("Đang load model...")
model = YOLO(MODEL_PATH)

# 2. Mở Video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Không thể mở video: {VIDEO_PATH}")
    exit()

print("Nhấn 'q' để thoát...")

while True:
    ret, frame = cap.read()
    if not ret:
        break # Hết video thì dừng

    # 3. Detect (Dùng device=0 nếu có GPU, không thì xóa đi để chạy CPU)
    results = model(frame, conf=CONF_THRESHOLD, device=0)

    # 4. Vẽ Bounding Box (Hàm plot() của YOLO tự vẽ rất đẹp)
    annotated_frame = results[0].plot()

    # (Tùy chọn) Resize nhỏ lại nếu video quá to để xem cho dễ
    display_frame = cv2.resize(annotated_frame, (960, 540))

    # 5. Hiển thị
    cv2.imshow("YOLOv8 Simple Detect", display_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()