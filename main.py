import cv2
import numpy as np
import onnxruntime as ort
import argparse

# ==============================
# CLI
# ==============================
parser = argparse.ArgumentParser(
    description="Blur specific face using YOLO + Face Embedding (NO dlib)"
)
parser.add_argument("video")
parser.add_argument("--target", required=True)
parser.add_argument("--yolo", default="yolov8n-face.onnx")
parser.add_argument("--embed", default="arcface_r100.onnx")
parser.add_argument("-o", "--output", default="output.mp4")
parser.add_argument("--threshold", type=float, default=1.0)
parser.add_argument("--padding", type=float, default=0.3)
args = parser.parse_args()

# ==============================
# 모델 로드
# ==============================
yolo = ort.InferenceSession(args.yolo, providers=["CPUExecutionProvider"])
embedder = ort.InferenceSession(args.embed, providers=["CPUExecutionProvider"])

# ==============================
# 얼굴 임베딩 함수
# ==============================
def get_embedding(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    emb = embedder.run(None, {embedder.get_inputs()[0].name: face})[0]
    emb = emb / np.linalg.norm(emb)
    return emb[0]

# ==============================
# 타겟 얼굴 임베딩
# ==============================
target_img = cv2.imread(args.target)
target_emb = get_embedding(target_img)

# ==============================
# 비디오 준비
# ==============================
cap = cv2.VideoCapture(args.video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    args.output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ==============================
# 메인 루프
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.resize(frame, (640, 640))
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    preds = yolo.run(None, {yolo.get_inputs()[0].name: blob})[0][0]

    for det in preds:
        conf = det[4]
        if conf < 0.5:
            continue

        cx, cy, bw, bh = det[:4]
        x1 = int((cx - bw / 2) * w / 640)
        y1 = int((cy - bh / 2) * h / 640)
        x2 = int((cx + bw / 2) * w / 640)
        y2 = int((cy + bh / 2) * h / 640)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        emb = get_embedding(face)
        dist = np.linalg.norm(emb - target_emb)

        # ==============================
        # 🔥 특정 얼굴만 블러
        # ==============================
        if dist < args.threshold:
            pad = int((x2 - x1) * args.padding)
            xx1 = max(0, x1 - pad)
            yy1 = max(0, y1 - pad)
            xx2 = min(w, x2 + pad)
            yy2 = min(h, y2 + pad)

            roi = frame[yy1:yy2, xx1:xx2]
            roi = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[yy1:yy2, xx1:xx2] = roi

    out.write(frame)

cap.release()
out.release()
print("✅ 완료:", args.output)
