import cv2
import numpy as np
from insightface.app import FaceAnalysis

# =====================
# 설정
# =====================
INPUT_VIDEO = "res\\input.mp4"
OUTPUT_VIDEO = "output_blur.mp4"
TARGET_IMAGE = "res\\target.jpg"

SIM_THRESHOLD = 0.4      # 이 값 이상이면 같은 사람
BLUR_KERNEL = (51, 51)   # 블러 강도 (홀수만 가능)

# =====================
# InsightFace 초기화
# =====================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# =====================
# Target embedding
# =====================
target_img = cv2.imread(TARGET_IMAGE)
target_faces = app.get(target_img)

if len(target_faces) == 0:
    raise RuntimeError("❌ target.jpg에서 얼굴을 찾지 못함")

target_emb = target_faces[0].embedding
target_emb = target_emb / np.linalg.norm(target_emb)  # ⭐ L2 normalize

# =====================
# 비디오 입출력 설정
# =====================
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("❌ 입력 영상 열기 실패")

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# =====================
# 메인 루프
# =====================
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)  # ⭐ L2 normalize

        sim = np.dot(emb, target_emb)

        if sim > SIM_THRESHOLD:
            x1, y1, x2, y2 = map(int, face.bbox)

            # 안전 클램프
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            blur = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
            frame[y1:y2, x1:x2] = blur

    out.write(frame)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"처리 중... {frame_idx} frames")

# =====================
# 정리
# =====================
cap.release()
out.release()

print("✅ 완료:", OUTPUT_VIDEO)