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


rows = []
for name, doc in lesson['documents'].items():
    for answer in doc['mock_vlm']['responses']:
        # 문자열 포함 검사는 1차 점검일 뿐 의미상 올바른 답임을 증명하지 않습니다.
        # 해설 A04
                # VLM이 답변의 근거로 제시한 문자열(Evidence)이 실제 OCR 원문에 존재하는지 확인
        #
        # answer['evidence'] : VLM이 답변의 근거로 제시한 텍스트
        # doc['ocr']['text'] : OCR을 통해 문서 이미지에서 추출한 전체 텍스트
        #
        # in 연산자:
        #   → 근거 문자열이 OCR 원문에 포함되어 있으면 True
        #   → 포함되어 있지 않으면 False
        #
        # evidence_found:
        #   → True  : VLM이 제시한 근거가 OCR 원문에 존재함
        #   → False : 근거가 OCR 원문에서 확인되지 않음
        #
        # 목적:
        # VLM이 제시한 근거를 OCR 결과와 대조하여
        # 답변의 근거 존재 여부를 검증하고 환각(Hallucination) 가능성을 점검

        evidence_found = answer['evidence'] in doc['ocr']['text']
        # 해설 끝 A04
        
        
        rows.append({'document_id':name, 'backend':'mock', **answer, 'evidence_found':evidence_found, 'review':not evidence_found})
table = pd.DataFrame(rows)
table.to_csv(out / 'p04_mock_review.csv', index=False, encoding='utf-8-sig')
print('mock 필드:', len(table), '근거 문자열 발견:', int(table['evidence_found'].sum()))
print('모의 응답의 평가 연습이며 실제 VLM 정확도가 아닙니다.')
