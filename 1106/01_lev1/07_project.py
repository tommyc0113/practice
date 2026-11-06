"""실습 07~08. 실제 모델의 이미지 탐지와 영상 추적을 연결합니다."""
from pathlib import Path
import cv2
import torch
import pandas as pd
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
torch.set_num_threads(1)

# YOLO 가중치 모델 불러오기
model = YOLO(str(BASE / "data/demo.pt"))

# 이미지 탐지 결과 저장 리스트
rows = []

# valid 이미지 폴더의 객체 탐지 실행
for r in model.predict(
    source=str(BASE / "data/images/val"),
    conf=0.25,
    imgsz=160,
    device=0,
    verbose=False
):
    for box, cls, score in zip(
        r.boxes.xyxy.cpu().tolist(),
        r.boxes.cls.cpu().tolist(),
        r.boxes.conf.cpu().tolist()
    ):
        rows.append([Path(r.path).name, int(cls), score, *box])

# 이미지 탐지 결과 CSV 저장
pd.DataFrame(
    rows,
    columns=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"]
).to_csv(BASE / "p07_predictions.csv", index=False)

# 새로운 영상 추적을 위한 모델 인스턴스 생성
tracker = YOLO(str(BASE / "data/demo.pt"))

# 입력 영상 정보 확인
cap = cv2.VideoCapture(str(BASE / "data/demo.avi"))
fps = cap.get(cv2.CAP_PROP_FPS)
size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
cap.release()

# 추적 결과를 저장할 영상 파일 생성
writer = cv2.VideoWriter(
    str(BASE / "p07_tracked.avi"),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    size
)

# 추적 결과 저장 리스트
tracks = []
frame_rows = []

# >>> B08
# demo.avi 영상을 ByteTrack으로 추적합니다.
results = tracker.track(
    source=str(BASE / "data/demo.avi"),  # 추적할 영상 파일 경로
    tracker="bytetrack.yaml",            # ByteTrack 추적 알고리즘
    stream=True,                         # 프레임을 순차적으로 처리
    persist=False,                       # 새로운 영상 추적 상태로 시작
    conf=0.25,                            # 객체 탐지 신뢰도 기준 25%
    imgsz=160,                            # 모델 입력 이미지 크기
    device="cpu",                         # CPU 사용
    verbose=False                         # 상세 로그 출력 생략
)
# <<< B08

# 영상 프레임별 추적 결과 처리
for frame, result in enumerate(results):

    # 객체 탐지 결과를 영상 프레임에 표시하고 저장
    writer.write(result.plot())

    # 프레임 번호와 탐지 개수 저장
    frame_rows.append([frame, len(result.boxes)])

    # 추적 ID가 존재하는 경우 객체 정보 저장
    if result.boxes.id is not None:
        for box, cls, score, tid in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.id.cpu().tolist()
        ):
            tracks.append([
                frame,
                int(tid),
                int(cls),
                score,
                *box
            ])

# 영상 저장 종료
writer.release()

# >>> B09
# 추적 결과를 DataFrame으로 변환하고 CSV 파일로 저장
pd.DataFrame(
    tracks,
    columns=[
        "frame",       # 프레임 번호
        "track_id",    # 객체 추적 ID
        "class_id",    # 객체 클래스 번호
        "confidence",  # 객체 탐지 신뢰도
        "x1",          # 바운딩 박스 왼쪽 위 x 좌표
        "y1",          # 바운딩 박스 왼쪽 위 y 좌표
        "x2",          # 바운딩 박스 오른쪽 아래 x 좌표
        "y2"           # 바운딩 박스 오른쪽 아래 y 좌표
    ]
).to_csv(
    BASE / "p07_tracks.csv",
    index=False
)
# <<< B09

# 프레임별 탐지 개수 CSV 저장
pd.DataFrame(
    frame_rows,
    columns=["frame", "detections"]
).to_csv(BASE / "p07_frames.csv", index=False)

# 실행 결과 출력
print("입력 fps/size:", fps, size, "처리 Frame:", len(frame_rows))
print("추적 객체 수:", len(tracks))
print("실습 08: 설정을 바꾸지 말고 다시 실행해 Frame 수·CSV·영상을 확인하세요.")