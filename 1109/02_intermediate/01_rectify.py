from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cv2
base = Path(__file__).resolve().parent
data = base / 'data'
out = base / 'outputs'
out.mkdir(exist_ok=True)
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

image = cv2.imread(str(data / 'perspective.png'))
corners = np.float32(lesson['perspective']['corners'])  # TL, TR, BR, BL 제공
size = (1000, 700)  # width, height
matrix = cv2.getPerspectiveTransform(corners, np.float32([[0,0],[999,0],[999,699],[0,699]]))
# TODO A01
# 원근 변환(Perspective Transformation)을 적용하여 기울어진 문서 이미지를 보정
#
# cv2.warpPerspective():
#   → 원근 변환 행렬을 사용하여 이미지의 좌표를 변환
#   → 비스듬히 촬영된 문서를 정면에서 본 것처럼 펼치는 데 사용
#
# image  : 원본 이미지 (기울어지거나 원근 왜곡이 있는 이미지)
# matrix : 3×3 원근 변환 행렬 (cv2.getPerspectiveTransform() 등으로 계산)
# size   : 변환 후 출력 이미지 크기 (width, height)
#
# rectified:
#   → 원근 왜곡을 보정한 결과 이미지
#   → 문서 영역을 직사각형으로 정렬하여 OCR 인식에 유리한 형태로 변환
rectified = cv2.warpPerspective(image, matrix, size)
# TODO 끝 A01
gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
cv2.imwrite(str(out / 'p01_rectified.png'), rectified)
cv2.imwrite(str(out / 'p01_adaptive.png'), adaptive)
print('before:', image.shape, 'after:', rectified.shape)
print('알려진 모서리로 보정했습니다. 모서리 자동 검출이 아닙니다.')
