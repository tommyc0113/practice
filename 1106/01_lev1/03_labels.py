"""실습 03. 정상 이미지 6장과 YOLO TXT 라벨을 읽습니다."""
from pathlib import Path
import cv2
BASE = Path(__file__).resolve().parent
names = ["square", "circle", "triangle"]
for image_path in sorted((BASE / "data/images/train").glob("*.png"))[:6]:
    image = cv2.imread(str(image_path))
    label_path = BASE / "data/labels/train" / (image_path.stem + ".txt")
    # >>> B04
    lines = label_path.read_text(encoding="utf-8").splitlines()
    # <<< B047 
    print(image_path.name, "shape:", image.shape, "boxes:", len(lines))
    for line in lines:
        cls, cx, cy, width, height = map(float, line.split())
        print(" ", names[int(cls)], "normalized:", cx, cy, width, height)

"""
line.split(): 한 줄을 공백 기준으로 분리
map(float, ...): 분리한 문자열을 실수형으로 변환
cls: 클래스 번호
cx, cy: 중심 좌표
"""