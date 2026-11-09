
"""
실습 05: OCR 기반 문서 필드 추출 정확도 평가

[실습 목적]
OCR로 추출한 문서 텍스트에서 날짜·공급업체·총금액을 추출하고,
실제 정답(Ground Truth)과 비교하여 필드별 정확 일치율을 평가합니다.

[주요 동작]
1. lesson.json에서 OCR 텍스트 및 정답 데이터 로딩
2. 정규표현식으로 날짜·공급업체·총금액 추출
3. 추출값과 실제 정답을 필드별로 비교
4. 결과를 CSV로 저장하고 전체 일치 건수 출력

※ 총 5개 문서 × 3개 필드 = 15개 항목을 평가합니다.
※ 정답은 평가 단계에서만 사용합니다.
"""

from pathlib import Path
import json
import re
import pandas as pd

# ============================================================
# 1. 파일 경로 및 데이터 로딩
# ============================================================

# 현재 Python 파일이 위치한 디렉터리
base = Path(__file__).resolve().parent

# OCR 텍스트와 실제 정답이 포함된 JSON 파일 읽기
lesson = json.loads(
    (base / 'lesson.json').read_text(encoding='utf-8')
)

# 문서별 평가 결과 저장 리스트
rows = []


# ============================================================
# 2. 문서별 OCR 텍스트에서 필드 추출
# ============================================================

# lesson.json에 저장된 문서를 하나씩 순회
for name, doc in lesson['documents'].items():

    # OCR로 인식한 문서 전체 텍스트
    text = doc['ocr']['text']

    # 정규표현식(Regex)을 이용하여 주요 필드 추출
    predicted = {

        # DATE: YYYY-MM-DD 형식에서 날짜 추출
        # \d{4}: 연도 4자리, \d{2}: 월·일 2자리
        # group(1): 첫 번째 괄호 안의 값 반환
        'issued_at': re.search(
            r'DATE: (\d{4}-\d{2}-\d{2})',
            text
        ).group(1),

        # SUPPLIER: 뒤에 있는 업체명 추출
        # (.+): 줄바꿈 전까지 하나 이상의 문자 추출
        'supplier': re.search(
            r'SUPPLIER: (.+)',
            text
        ).group(1),

        # TOTAL: 뒤에 있는 숫자 추출
        # \d+: 연속된 숫자
        # int(): 추출된 문자열을 정수로 변환
        'total_amount': int(
            re.search(
                r'TOTAL: (\d+)',
                text
            ).group(1)
        )
    }

    # ========================================================
    # 3. OCR 추출값과 실제 정답 비교
    # ========================================================

    # 날짜, 공급업체, 총금액을 각각 평가
    for field in ['issued_at', 'supplier', 'total_amount']:

        # predicted[field]: OCR에서 추출한 예측값
        # doc['truth'][field]: 실제 정답(Ground Truth)
        #
        # 두 값이 정확히 일치하면 True, 다르면 False
        # 예: "2026-11-09" == "2026-11-09" → True
        # 예: 15000 == 150000 → False
        correct = predicted[field] == doc['truth'][field]

        # 문서 ID, 필드명, 예측값, 정답 일치 여부 기록
        rows.append({
            'document_id': name,
            'field': field,
            'predicted': predicted[field],
            'correct': correct
        })


# ============================================================
# 4. 평가 결과를 CSV로 저장
# ============================================================

# 평가 결과를 DataFrame으로 변환
table = pd.DataFrame(rows)

# 문서별·필드별 정답 일치 여부 저장
# utf-8-sig: Excel에서 한글이 깨지는 문제를 줄임
table.to_csv(
    base / 'p05_field_scores.csv',
    index=False,
    encoding='utf-8-sig'
)


# ============================================================
# 5. 최종 평가 결과 출력
# ============================================================

# 전체 문서 수와 평가한 필드 수 출력
print(
    '문서 수:', len(lesson['documents']),
    '필드 수:', len(table)
)

# correct=True인 항목의 개수 계산
# Boolean 값은 합계 계산 시 True=1, False=0으로 처리
print(
    '필드 일치:',
    int(table['correct'].sum()),
    '/',
    len(table)
)

# 평가 데이터의 범위와 한계 안내
print(
    '합성 영어 문서의 기록 비교이며 일반 OCR·한국어 성능이 아닙니다.'
)