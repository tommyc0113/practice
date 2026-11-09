"""
실습 05: Mock VLM의 문서 필드 추출 결과 평가

[실습 목적]
Mock VLM이 문서에서 추출한 답변을 실제 정답(Ground Truth)과 비교하고,
답변의 근거가 OCR 원문에 존재하는지 검증합니다.

[주요 동작]
1. lesson.json에서 문서 및 Mock VLM 응답 불러오기
2. VLM 예측값과 실제 정답의 일치 여부(Exact Match) 평가
3. VLM이 제시한 근거(Evidence)가 OCR 원문에 존재하는지 확인
4. 평가 결과를 CSV 파일로 저장하고 전체 일치 건수 출력

[주의사항]
- 실제 VLM을 실행하지 않고 미리 준비된 Mock 응답을 사용합니다.
- 정답 일치와 근거 존재 여부는 서로 다른 평가 항목입니다.
- 실제 모델 추론을 수행하지 않으므로 추론 지연시간은 측정하지 않습니다.
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cv2

# ============================================================
# 1. 경로 설정 및 데이터 로딩
# ============================================================

# 현재 Python 파일의 디렉터리
base = Path(__file__).resolve().parent

# 입력 데이터와 출력 결과 저장 경로
data = base / 'data'
out = base / 'outputs'
out.mkdir(exist_ok=True)

# 문서별 OCR 결과, 정답, Mock VLM 응답이 포함된 JSON 파일 로딩
lesson = json.loads(
    (data / 'lesson.json').read_text(encoding='utf-8')
)


# ============================================================
# 2. 문서별 Mock VLM 응답 평가
# ============================================================

# 평가 결과를 저장할 리스트
rows = []

# 전체 문서를 하나씩 순회
for name, doc in lesson['documents'].items():

    # 현재 문서에 대한 Mock VLM 응답을 하나씩 확인
    for item in doc['mock_vlm']['responses']:

        # ----------------------------------------------------
        # 2-1. 예측값과 실제 정답 비교 (Exact Match)
        # ----------------------------------------------------

        # item['field'] : 평가 대상 필드 (예: issued_at, supplier)
        # item['answer']: Mock VLM이 추출한 예측값
        # doc['truth'] : 현재 문서의 실제 정답 데이터
        #
        # 예측값과 정답이 정확히 일치하면 True, 다르면 False
        # 문자열은 공백·대소문자 차이도 일치 여부에 영향을 줌
        exact = item['answer'] == doc['truth'][item['field']]

        # ----------------------------------------------------
        # 2-2. OCR 원문에서 답변 근거 확인 (Evidence Check)
        # ----------------------------------------------------

        # item['evidence'] : VLM이 답변의 근거로 제시한 문자열
        # doc['ocr']['text']: OCR로 추출한 전체 문서 텍스트
        #
        # in 연산자:
        # 근거 문자열이 OCR 원문에 존재하면 True, 없으면 False
        #
        # 주의: 문자열 존재 여부만 확인하며
        # 해당 근거가 답변을 올바르게 뒷받침하는지는 판단하지 않음
        grounded = item['evidence'] in doc['ocr']['text']

        # ----------------------------------------------------
        # 2-3. 평가 결과 기록
        # ----------------------------------------------------

        # exact          : 실제 정답과의 일치 여부
        # evidence_found : OCR 원문에서 근거 문자열 발견 여부
        # review         : 근거가 없으면 사람 검토 대상으로 표시
        #
        # backend='mock': 실제 VLM이 아닌 가상 응답임을 명시
        rows.append({
            'document_id': name,
            'field': item['field'],
            'backend': 'mock',
            'exact': exact,
            'evidence_found': grounded,
            'review': not grounded
        })


# ============================================================
# 3. 평가 결과 CSV 저장
# ============================================================

# 문서별·필드별 평가 결과를 DataFrame으로 변환
table = pd.DataFrame(rows)

# 평가 결과 저장
# utf-8-sig: Excel 등에서 한글이 깨지는 문제를 줄임
table.to_csv(
    out / 'p05_scores.csv',
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 4. 최종 평가 결과 출력
# ============================================================

# 전체 평가 항목 중 정답과 정확히 일치한 개수
# True=1, False=0으로 계산
print(
    '값 일치:',
    int(table['exact'].sum()),
    '/',
    len(table)
)

# 정답 일치(exact)와 OCR 근거 존재(evidence_found)를
# 동시에 만족하는 항목 개수 계산
#
# &: Pandas Boolean Series의 요소별 AND 연산
# 두 조건이 모두 True일 때만 1건으로 집계
print(
    '값과 문자열 근거:',
    int(
        (table['exact'] & table['evidence_found']).sum()
    ),
    '/',
    len(table)
)

# 실제 VLM 추론을 수행하지 않았으므로 지연시간 측정 불가
print('추론 지연시간: N/A (mock)')