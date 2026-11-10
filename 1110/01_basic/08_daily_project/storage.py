
"""
JSON 파일 저장 및 읽기 모듈
"""

import json
import threading

from pathlib import Path


# 현재 storage.py가 위치한 폴더
BASE_DIR = Path(__file__).resolve().parent

# 데이터 파일 경로
DATA_FILE = BASE_DIR / "data.json"


# 동시 접근 방지용 Lock
LOCK = threading.Lock()


# --------------------------------------------------
# JSON 데이터 읽기
# --------------------------------------------------

def read_data():

    # 데이터 파일이 없으면 기본 구조 생성
    if not DATA_FILE.exists():

        data = {
            "next_id": 1,
            "entries": []
        }

        write_data(data)

        return data

    # JSON 파일 읽기
    with open(DATA_FILE, "r", encoding="utf-8") as f:

        return json.load(f)


# --------------------------------------------------
# JSON 데이터 저장
# --------------------------------------------------

def write_data(data):

    with open(DATA_FILE, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )
