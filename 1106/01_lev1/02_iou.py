"""실습 02. 손으로 정한 예측 4개를 정답 3개와 비교합니다. YOLO 결과가 아닙니다."""
from pathlib import Path
import numpy as np
import pandas as pd
BASE = Path(__file__).resolve().parent
pred = pd.read_csv(BASE / "data/boxes_fixture.csv")  # confidence 내림차순 예시
truth = pd.read_csv(BASE / "data/boxes_gt.csv")
used = set()  # 하나의 정답은 한 번만 TP로 연결합니다.
rows = []
for p in pred.itertuples():
    overlaps = []
    for g in truth.itertuples():
        width = max(0, min(p.x2, g.x2) - max(p.x1, g.x1))
        height = max(0, min(p.y2, g.y2) - max(p.y1, g.y1))
        intersection = width * height
        union = (p.x2-p.x1)*(p.y2-p.y1) + (g.x2-g.x1)*(g.y2-g.y1) - intersection
        # >>> B02
        # 요구: intersection을 union으로 나누어 iou를 계산하세요.
        # 이 실습의 데이터는 모두 양의 면적이므로 union은 0이 아닙니다.
        # TODO B02: 위 요구사항의 핵심 코드를 작성하세요.
        # >>> B02
        iou = intersection / union
        # <<< B02        # <<< B02
        overlaps.append(iou)
    matched = False
    for j in np.argsort(-np.asarray(overlaps)):
        if truth.iloc[j].class_id != p.class_id or j in used:
            continue
        # >>> B03
        # 요구: 같은 클래스의 미사용 정답에 IoU>=0.5이면 연결하고 used에 기록하세요.
        pass  # TODO B03: 위 요구사항의 핵심 코드를 작성하세요.
        # >>> B03
        if overlaps[j] >= 0.5:
            used.add(j)
            matched = True
            break

        # <<< B03
    rows.append([p.class_id, p.confidence, round(max(overlaps), 4), "TP" if matched else "FP"])
table = pd.DataFrame(rows, columns=["class", "confidence", "max_iou", "result"])
print(table.to_string(index=False))
tp = len(used); fp = len(pred)-tp; fn = len(truth)-tp
print("TP/FP/FN:", tp, fp, fn)
print("Precision/Recall:", round(tp/len(pred), 4), round(tp/len(truth), 4))
table.to_csv(BASE / "p02_matching.csv", index=False, encoding="utf-8-sig")
"""
[실습 02] IoU 기반 객체 탐지 결과 분석
1. 주요 결과
- TP(정탐) 2건: 실제 정답 객체 3개 중 2개를 올바르게 탐지하였다.
- FP(오탐) 2건: 중복 탐지 1건과 정답 객체에 대응하지 않는 예측 1건이 발생하였다.
- FN(미탐) 1건: 실제 정답 객체 1개를 탐지하지 못하였다.
- Precision 50.00%: 전체 예측 4건 중 2건만 올바른 탐지로 인정되었다.
- Recall 66.67%: 실제 객체 3개 중 2개를 성공적으로 탐지하였다.

2. 결과 해석
IoU가 0.8162로 높은 예측도 이미 다른 예측과 연결된 정답 객체를 중복 탐지했기 때문에 FP로 처리되었다. 
이는 객체 탐지 평가에서 IoU뿐 아니라 클래스 일치 여부와 정답 객체의 중복 매칭 방지가 중요하다는 것을 보여준다.

3. 최종 결론
본 실습에서는 IoU 0.5 기준으로 Precision 50.00%, Recall 66.67%를 기록하였다. 
객체의 위치가 정확하더라도 중복 탐지나 미탐지가 발생하면 전체 탐지 성능이 저하될 수 있음을 확인하였다.
따라서 실제 객체 탐지 시스템에서는 IoU 기반 매칭과 함께 중복 탐지 억제(NMS), 
Confidence Threshold 조정 등을 통해 오탐과 미탐을 관리할 필요가 있다.
"""