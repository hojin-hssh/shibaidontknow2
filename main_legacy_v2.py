import cv2

# ==============================
# 얼굴 인식 모델 로드
# ==============================
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ==============================
# 입력 / 출력 영상
# ==============================
input_video = "input.mp4"
output_video = "output_blur.mp4"

cap = cv2.VideoCapture(input_video)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# ==============================
# 🔥 [개선 1] 얼굴 캐시 (프레임 유지)
# ==============================
last_faces = []        # 이전 프레임에서 인식된 얼굴
FACE_MEMORY = 10       # 인식 실패해도 유지할 프레임 수
memory_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # 🔥 [개선 2] 그레이스케일 + 히스토그램 평활화
    # → 어두운 환경에서 인식률 상승
    # ==============================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ==============================
    # 🔥 [개선 3] 얼굴 탐지 민감도 조정
    # ==============================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,   # ⬇️ 더 촘촘하게 탐색
        minNeighbors=3,     # ⬇️ 판정 기준 완화
        minSize=(20, 20)    # ⬇️ 작은 얼굴 허용
    )

    # ==============================
    # 🔥 [개선 4] 탐지 실패 시 이전 얼굴 유지
    # ==============================
    if len(faces) > 0:
        last_faces = faces
        memory_counter = FACE_MEMORY
    else:
        if memory_counter > 0:
            faces = last_faces
            memory_counter -= 1

    # ==============================
    # 얼굴 블러 처리
    # ==============================
    for (x, y, w, h) in faces:

        # 🔥 [개선 5] 얼굴 영역 padding (30%)
        # → 고개 회전 / 흔들림 대응
        padding = int(0.3 * w)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face = frame[y1:y2, x1:x2]

        # 블러 처리
        face_blur = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y1:y2, x1:x2] = face_blur

    out.write(frame)

cap.release()
out.release()

print("✅ 얼굴 블러 처리 완료:", output_video)
