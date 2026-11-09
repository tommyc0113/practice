"""실습 02: Tesseract를 실제 호출합니다. 엔진과 eng 언어팩이 필요합니다."""
from pathlib import Path
import os
import pandas as pd
import pytesseract

base = Path(__file__).resolve().parent
# Windows에서 PATH 설정이 어려우면 환경변수 TESSERACT_CMD에 exe 경로를 지정합니다.
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
paths = [str(base / f'doc{i:02d}.png') for i in range(1, 4)]
input_list = base / 'p02_inputs.txt'
input_list.write_text('\n'.join(paths), encoding='utf-8')
# 여러 페이지를 목록으로 전달해 엔진을 한 번 호출합니다.
# 해설 B02
# Tesseract OCR을 사용하여 이미지에서 문자(Text)를 인식하고 상세 정보를 추출
#
# pytesseract.image_to_data():
#   → 이미지의 문자 인식 결과를 단어 단위로 반환
#   → 인식된 텍스트뿐 아니라 위치 좌표, 크기, 신뢰도(Confidence)도 제공
#
# str(input_list) :
#   → OCR을 수행할 이미지 파일 경로를 문자열로 전달
#   → input_list가 실제 이미지 파일 경로라고 가정
#
# lang='eng' :
#   → 영어 인식 모델 사용 (한글은 'kor', 한영 혼합은 'kor+eng')
#
# config='--psm 6' :
#   → Page Segmentation Mode 6
#   → 이미지 전체를 하나의 균일한 텍스트 블록으로 가정하고 문자 인식
#   → 문단 형태의 문서를 인식할 때 적합
#
# output_type=pytesseract.Output.DATAFRAME :
#   → OCR 결과를 Pandas DataFrame 형태로 반환
#   → text, conf, left, top, width, height 등의 컬럼 포함
#
# words :
#   → 단어별 인식 텍스트, Bounding Box, 신뢰도 등의 OCR 분석 결과
words = pytesseract.image_to_data(
    str(input_list),
    lang='eng',
    config='--psm 6',
    output_type=pytesseract.Output.DATAFRAME
)
# 해설 끝 B02
words = words.dropna(subset=['text'])
words = words[words['conf'] >= 0]
words.to_csv(base / 'p02_words.csv', index=False, encoding='utf-8-sig')
for page, rows in words.groupby('page_num', sort=True):
    text = ' '.join(rows['text'].astype(str))
    print('page:', page, 'words:', len(rows), 'mean confidence:', round(rows['conf'].mean(), 2))
    print(text)
print('backend: live Tesseract; confidence는 정답 확률이 아닙니다.')