from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cv2

# 현재 파이썬 파일이 위치한 폴더 경로를 가져옵니다.
base = Path(__file__).resolve().parent

# 입력 데이터 폴더와 결과 저장 폴더 경로를 지정합니다.
data = base / 'data'
out = base / 'outputs'

# 결과 저장 폴더가 없으면 생성합니다.
out.mkdir(exist_ok=True)

# lesson.json 파일을 읽어 JSON 데이터를 가져옵니다.
lesson = json.loads((data / 'lesson.json').read_text(encoding='utf-8'))

# 전체 문서 중 doc01 문서 정보를 가져옵니다.
doc = lesson['documents']['doc01']

# OCR에서 인식한 단어와 좌표 정보를 가져옵니다.
words = doc['ocr']['tokens']

# 추출한 필드 정보를 저장할 빈 딕셔너리를 생성합니다.
fields = {}

# 날짜, 공급자, 총금액을 순서대로 처리합니다.
for key, label in [('issued_at','DATE:'), ('supplier','SUPPLIER:'), ('total_amount','TOTAL:')]:

    # OCR 단어 중 현재 라벨과 일치하는 단어를 찾습니다.
    anchor = next(w for w in words if w['text'] == label)

    # TODO A03 시작

    # 같은 block/paragraph/line이고 라벨 오른쪽에 있는 단어만 선택합니다.
    # 해설 A03
        # OCR로 인식한 단어 목록(words)에서 특정 기준 단어(anchor)의
    # 오른쪽에 위치한 단어들만 추출합니다.
    #
    # words  : OCR로 인식한 전체 단어 정보 목록
    # anchor : 위치를 판단할 기준 단어 (예: "DATE:", "TOTAL:")
    #
    # ① w['block'] == anchor['block']
    #    → 기준 단어와 동일한 텍스트 블록에 속하는지 확인
    #
    # ② w['paragraph'] == anchor['paragraph']
    #    → 기준 단어와 동일한 문단에 속하는지 확인
    #
    # ③ w['line'] == anchor['line']
    #    → 기준 단어와 동일한 줄에 속하는지 확인
    #
    # ④ w['bbox'][0] >= anchor['bbox'][2]
    #    → 현재 단어의 왼쪽 X좌표가 기준 단어의 오른쪽 X좌표 이상인지 확인
    #    → 즉, 기준 단어의 오른쪽에 있는 단어만 선택
    #
    # bbox = [x1, y1, x2, y2]
    # x1: 왼쪽 좌표, x2: 오른쪽 좌표
    #
    # 최종 결과:
    # 기준 단어와 같은 블록·문단·줄에 있으면서,
    # 기준 단어의 오른쪽에 있는 단어들을 selected 리스트에 저장

    selected = [
        w for w in words
        if (
            w['block'],
            w['paragraph'],
            w['line']
        ) == (
            anchor['block'],
            anchor['paragraph'],
            anchor['line']
        )
        and w['bbox'][0] >= anchor['bbox'][2]
    ]
    # TODO 끝 A03

    # 선택된 단어들을 공백으로 연결하여 하나의 문자열로 만듭니다.
    raw = ' '.join(w['text'] for w in selected)

    # 총금액은 정수형으로 변환하고 날짜와 공급자는 문자열로 유지합니다.
    value = int(raw.replace(',', '').replace('₩', '')) if key == 'total_amount' else raw

    # 선택된 단어들의 좌표를 합쳐 전체 영역의 Bounding Box를 계산합니다.
    box = [
        min(w['bbox'][0] for w in selected),
        min(w['bbox'][1] for w in selected),
        max(w['bbox'][2] for w in selected),
        max(w['bbox'][3] for w in selected)
    ]

    # 추출한 값, 신뢰도, 좌표 정보를 딕셔너리에 저장합니다.
    fields[key] = {
        'raw': raw,
        'value': value,
        'confidence': min(w['conf'] for w in selected),
        'bbox': box,
        'coordinate_space': 'original_pixels'
    }

# 추출 결과를 JSON 파일로 저장합니다.
(out / 'p03_evidence.json').write_text(
    json.dumps(fields, ensure_ascii=False, indent=2),
    encoding='utf-8'
)

# 추출된 필드값을 화면에 출력합니다.
print({k: v['value'] for k, v in fields.items()})