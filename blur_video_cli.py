import cv2
import numpy as np
import argparse
from insightface.app import FaceAnalysis

# =====================
# CLI 파서
# =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Blur specific person in a video using InsightFace"
    )
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    parser.add_argument("--target", required=True, help="타겟 얼굴 이미지")
    parser.add_argument("--out", default="output_blur.mp4", help="출력 영상 경로")
    parser.add_argument("--sim", type=float, default=0.4, help="similarity threshold")
    parser.add_argument("--blur", type=int, default=51, help="블러 커널 크기 (홀수)")
    parser.add_argument("--det-size", type=int, default=640, help="얼굴 검출 해상도")

    return parser.parse_args()


# =====================
# 메인
# =====================
def main():
    args = parse_args()

    if args.blur % 2 == 0:
        raise ValueError("❌ --blur 값은 반드시 홀수여야 합니다")

    # =====================
    # InsightFace 초기화
    # =====================
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    # =====================
    # Target embedding
    # =====================
    target_img = cv2.imread(args.target)
    if target_img is None:
        raise RuntimeError("❌ target 이미지 로드 실패")

    target_faces = app.get(target_img)
    if len(target_faces) == 0:
        raise RuntimeError("❌ target 이미지에서 얼굴을 찾지 못함")

    target_emb = target_faces[0].embedding
    target_emb = target_emb / np.linalg.norm(target_emb)  # L2 normalize

    # =====================
    # 비디오 입출력
    # =====================
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("❌ 입력 영상 열기 실패")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

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
            emb = emb / np.linalg.norm(emb)

            sim = np.dot(emb, target_emb)

            if sim > args.sim:
                x1, y1, x2, y2 = map(int, face.bbox)

                # 클램프
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                blur = cv2.GaussianBlur(
                    roi,
                    (args.blur, args.blur),
                    0
                )
                frame[y1:y2, x1:x2] = blur

        out.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"처리 중... {frame_idx} frames")

    cap.release()
    out.release()

    print("✅ 완료:", args.out)


if __name__ == "__main__":
    main()