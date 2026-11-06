"""실습 07~08. Tracker 두 종류를 독립 인스턴스로 비교합니다."""
from pathlib import Path
from time import perf_counter
import cv2,torch
import pandas as pd
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True);torch.set_num_threads(1)
summaries=[]
for name in ["bytetrack.yaml","botsort.yaml"]:
    model=YOLO(str(BASE/"data/demo.pt"))  # 비교마다 상태가 새로 시작됩니다.
    rows=[];frame_counts=[];start=perf_counter()
    writer=cv2.VideoWriter(str(out/(Path(name).stem+".avi")),cv2.VideoWriter_fourcc(*"MJPG"),10,(160,128))
    # >>> A07
    # 요구: demo.avi를 현재 name의 tracker로 track 하세요. 반환은 stream generator입니다.
# >>> A07
# demo.avi 영상을 현재 반복문의 tracker 설정으로 추적합니다.
    results = model.track(
        source=str(BASE / "data/demo.avi"),  # 추적할 영상 경로
        tracker=name,                        # bytetrack 또는 botsort 적용
        stream=True,                         # 프레임을 순차적으로 반환
        persist=False,                       # 새로운 영상 추적 상태로 시작
        conf=0.25,                            # 객체 탐지 신뢰도 기준
        imgsz=160,                            # 모델 입력 이미지 크기
        device=0,                         # GPU 사용
        verbose=False                         # 상세 로그 출력 생략
    )
 # <<< A07
    for frame,r in enumerate(results):
        writer.write(r.plot());frame_counts.append([frame,len(r.boxes)])
        if r.boxes.id is not None:
            for box,cls,score,tid in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.id.cpu().tolist()):
                rows.append([frame,int(tid),int(cls),score,*box])
    writer.release();seconds=perf_counter()-start
    # >>> A08
    # 요구: tracker별 CSV에 frame·track_id·class_id·confidence·xyxy를 저장하세요.
# >>> A08
# 현재 Tracker의 추적 결과를 DataFrame으로 변환합니다.
    pd.DataFrame(
        rows,
        columns=[
            "frame",       # 프레임 번호
            "track_id",    # 객체 추적 ID
            "class_id",    # 객체 클래스 번호
            "confidence",  # 탐지 신뢰도
            "x1",          # 바운딩 박스 왼쪽 위 x 좌표
            "y1",          # 바운딩 박스 왼쪽 위 y 좌표
            "x2",          # 바운딩 박스 오른쪽 아래 x 좌표
            "y2"           # 바운딩 박스 오른쪽 아래 y 좌표
        ]
    ).to_csv(
        out / (Path(name).stem + "_tracks.csv"),  # Tracker 이름으로 파일 구분
        index=False                               # 인덱스 열 제외
    )
    # <<< A08
    pd.DataFrame(frame_counts,columns=["frame","detections"]).to_csv(out/(Path(name).stem+"_frames.csv"),index=False)
    summaries.append(dict(tracker=name,frames=len(frame_counts),track_rows=len(rows),seconds=seconds,throughput=len(frame_counts)/seconds,scope="decode+infer+track+plot+encode; first-run cost included",id_switch="별도 GT 매칭 필요"))
r=pd.DataFrame(summaries);r.to_csv(out/"tracker_comparison.csv",index=False);print(r.to_string(index=False))
