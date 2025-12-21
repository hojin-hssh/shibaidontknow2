import cv2
import numpy as np
import onnxruntime as ort

# =========================
# 설정
# =========================
VIDEO_PATH = "input.mp4"          # 테스트할 영상
YOLO_ONNX = "yolov8n-face.onnx"   # 얼굴 전용 YOLO ONNX
INPUT_SIZE = 640
CONF_TH = 0.01                    # 디버그용 (아주 낮게)

# =========================
# Letterbox 함수
# =========================
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))
    pad_top = (new_shape - nh) // 2
    pad_left = (new_shape - nw) // 2

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top+nh, pad_left:pad_left+nw] = img_resized

    return canvas, scale, pad_left, pad_top

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# =========================
# YOLO 세션 로드
# =========================
yolo_sess = ort.InferenceSession(
    YOLO_ONNX,
    providers=["CPUExecutionProvider"]
)
yolo_input_name = yolo_sess.get_inputs()[0].name

print("YOLO input shape:", yolo_sess.get_inputs()[0].shape)

# =========================
# 비디오 열기
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("비디오를 열 수 없음")

# =========================
# 메인 루프
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------
    # YOLO 입력 전처리
    # -------------------------
    img, scale, pad_x, pad_y = letterbox(frame, INPUT_SIZE)

    blob = img.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))   # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)    # [1,3,640,640]

    # -------------------------
    # YOLO 추론
    # -------------------------
    outputs = yolo_sess.run(None, {yolo_input_name: blob})
    out = outputs[0]

    print("YOLO output shape:", out.shape)
    print("YOLO max conf:", out[..., 4].max())

    # -------------------------
    # 박스 파싱 (가정: [N,6] or [1,N,6])
    # x1,y1,x2,y2,conf,cls
    # -------------------------
    if out.ndim == 3:
        out = out[0]

    for det in out:
        cx, cy, w, h, raw_conf = det[:5]
    
        conf = sigmoid(raw_conf)
        if conf < 0.3:
            continue
        
        # YOLOv8: center -> corner (640 기준)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
    
        # letterbox 복원
        x1 = int((x1 - pad_x) / scale)
        y1 = int((y1 - pad_y) / scale)
        x2 = int((x2 - pad_x) / scale)
        y2 = int((y2 - pad_y) / scale)
    
        # clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
    
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("YOLO FACE DEBUG", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
