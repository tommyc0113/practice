"""실습 01. 픽셀 xyxy와 YOLO 정규화 중심·크기를 왕복합니다."""
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
table=pd.read_csv(BASE/"data/boxes_gt.csv")
b=table[["x1","y1","x2","y2"]].to_numpy(float);width,height=160,128
# >>> A01
# 요구: 중심 x/y와 폭/높이를 계산하고 각 축 크기로 정규화하세요.
# 해설: 정규화 순서는 cx cy w h, 이미지 shape 순서 H W와 다릅니다.

# Bounding Box 좌표를 YOLO 형식으로 변환
#
# 기존 좌표 b: [x1, y1, x2, y2]
# x1, y1: Bounding Box의 왼쪽 위 좌표
# x2, y2: Bounding Box의 오른쪽 아래 좌표
#
# YOLO 좌표: [center_x, center_y, box_width, box_height]
# 모든 좌표는 이미지 전체 크기를 기준으로 0~1 범위로 정규화
#
# ① center_x = (x1 + x2) / 2 / width
#    → Bounding Box의 가로 중심 좌표를 이미지 너비로 나눔
#
# ② center_y = (y1 + y2) / 2 / height
#    → Bounding Box의 세로 중심 좌표를 이미지 높이로 나눔
#
# ③ box_width = (x2 - x1) / width
#    → Bounding Box의 가로 길이를 이미지 너비로 나눔
#
# ④ box_height = (y2 - y1) / height
#    → Bounding Box의 세로 길이를 이미지 높이로 나눔
#
# np.column_stack(): 4개 계산 결과를 열 방향으로 합쳐
#                    [N, 4] 형태의 YOLO 좌표 배열 생성
yolo = np.column_stack((
    (b[:, 0] + b[:, 2]) / (2 * width),
    (b[:, 1] + b[:, 3]) / (2 * height),
    (b[:, 2] - b[:, 0]) / width,
    (b[:, 3] - b[:, 1]) / height
))
pixel=yolo*np.array([width,height,width,height])
restored=np.column_stack((pixel[:,0]-pixel[:,2]/2,pixel[:,1]-pixel[:,3]/2,pixel[:,0]+pixel[:,2]/2,pixel[:,1]+pixel[:,3]/2))
print("최대 왕복 오차(px):", float(np.max(np.abs(b-restored))))
pd.DataFrame(yolo,columns=["cx","cy","w","h"]).to_csv(out/"yolo_coordinates.csv",index=False)
