import cv2

# 얼굴 인식 모델 로드
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# 입력 / 출력 영상
input_video = "input.mp4"
output_video = "output_blur.mp4"

cap = cv2.VideoCapture(input_video)

# 영상 정보
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 탐지
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        # 얼굴 블러 처리
        face_blur = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y:y+h, x:x+w] = face_blur

    out.write(frame)

cap.release()
out.release()

print("✅ 얼굴 블러 처리 완료:", output_video)
