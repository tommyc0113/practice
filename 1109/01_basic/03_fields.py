"""실습 03: 실제 OCR 기록에서 필요한 3개 필드를 추출합니다. 정상 자료만 사용합니다."""
from pathlib import Path
import json
import re

base = Path(__file__).resolve().parent
docs = json.loads((base / 'lesson.json').read_text(encoding='utf-8'))['documents']
records = []
for name in ['doc01', 'doc02', 'doc03']:
    text = docs[name]['ocr']['text']
# TODO B03
    # ============================================================
    # 1. 발행 날짜(DATE) 추출
    # ============================================================

    # re.search(): OCR 텍스트에서 정규표현식과 일치하는 문자열 검색
    #
    # ^DATE:       : 줄의 시작이 "DATE: "인 부분 검색
    # \d{4}        : 숫자 4자리 (연도 YYYY)
    # \d{2}        : 숫자 2자리 (월 MM, 일 DD)
    # $            : 줄의 끝을 의미
    # re.MULTILINE : 여러 줄의 텍스트에서 각 줄의 시작(^)과 끝($)을 인식
    # .group(1)    : 첫 번째 괄호 (...)에 해당하는 날짜 문자열 추출
    #
    # 예: "DATE: 2026-11-09" → "2026-11-09"
    issued_at = re.search(
        r'^DATE: (\d{4}-\d{2}-\d{2})$',
        text,
        re.MULTILINE
    ).group(1)


    # ============================================================
    # 2. 공급업체(SUPPLIER) 추출
    # ============================================================

    # ^SUPPLIER: : "SUPPLIER: "로 시작하는 줄 검색
    # (.+)       : 줄 끝까지 하나 이상의 문자 추출 (공급업체 이름)
    # .group(1)  : 공급업체 이름만 반환
    #
    # 예: "SUPPLIER: ABC Company" → "ABC Company"
    supplier = re.search(
        r'^SUPPLIER: (.+)$',
        text,
        re.MULTILINE
    ).group(1)


    # ============================================================
    # 3. 총금액(TOTAL) 추출
    # ============================================================

    # ^TOTAL:    : "TOTAL: "로 시작하는 줄 검색
    # (\d+)      : 숫자가 1개 이상 연속된 부분 추출
    # .group(1)  : 금액에 해당하는 숫자 문자열 반환
    # int()      : 문자열로 추출한 금액을 정수(int)로 변환
    #
    # 예: "TOTAL: 150000" → 150000 (정수)
    total_amount = int(
        re.search(
            r'^TOTAL: (\d+)$',
            text,
            re.MULTILINE
        ).group(1)
    )

# TODO 끝 B04
    records.append({'document_id': name, 'issued_at': issued_at, 'supplier': supplier, 'total_amount': total_amount, 'currency': 'KRW', 'ocr_source': 'recorded Tesseract', 'raw_text': text})
    print(name, issued_at, supplier, total_amount)
(base / 'p03_fields.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
