"""실습 05. 가상 기록 읽기. 이 표는 실제 모델 성능이 아닙니다."""
from pathlib import Path
import pandas as pd
BASE = Path(__file__).resolve().parent
history = pd.read_csv(BASE / "data/metrics_example.csv")
# >>> B06
# 요구: map50_95가 가장 큰 행을 선택하세요. idxmax는 첫 최대값의 index를 반환합니다.
best = history.loc[history["map50_95"].idxmax()]
# <<< B06
print("자료 종류: 읽기 연습용 가상 기록 — 실제 YOLO 성능 아님")
print("선택 epoch:", int(best["epoch"]))
print("mAP50:", best["map50"], "mAP50-95:", best["map50_95"])
print("mAP는 accuracy가 아닙니다. train이 아니라 valid의 위치·분류 평가 기록입니다.")

"""
.idxmax() : 가장 큰 값의 행 인덱스 반환
.loc[] : 해당 인덱스의 행 전체 선택
best : 최고 성능 기록이 저장된 행
"""