"""실습 02. 고전 NMS 원리 실험. YOLO26 기본 추론에 NMS가 있다고 가정하지 않습니다."""
from pathlib import Path
import torch
import pandas as pd
from torchvision.ops import box_iou, batched_nms
BASE=Path(__file__).resolve().parent
out=BASE/"outputs";out.mkdir(exist_ok=True)
p=pd.read_csv(BASE/"data/boxes_fixture.csv");g=pd.read_csv(BASE/"data/boxes_gt.csv")
boxes=torch.tensor(p[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
scores=torch.tensor(p.confidence.to_numpy(),dtype=torch.float32)
classes=torch.tensor(p.class_id.to_numpy(),dtype=torch.int64)
gb=torch.tensor(g[["x1","y1","x2","y2"]].to_numpy(),dtype=torch.float32)
rows=[]
for conf,nms_iou in [(0.25,0.5),(0.5,0.5),(0.25,0.95)]:
    ids=torch.where(scores>=conf)[0]
    # >>> A02
    # 요구: 같은 클래스끼리만 중복을 제거하는 batched_nms를 적용하세요.
    # 반환 index는 ids로 줄인 배열 기준이므로 다시 원본 index로 바꿉니다.
   # Batched NMS(Non-Maximum Suppression): 중복 Bounding Box 제거
    #
    # 목적:
    # 동일한 객체를 여러 Bounding Box가 중복 탐지했을 때,
    # Confidence Score가 높은 박스를 우선 남기고 중복된 박스를 제거합니다.
    #
    # boxes[ids]   : NMS를 적용할 Bounding Box 좌표 [x1, y1, x2, y2]
    # scores[ids]  : 각 Bounding Box의 Confidence Score
    # classes[ids] : 각 Bounding Box의 클래스 ID
    # nms_iou      : 중복 판단에 사용할 IoU 임계값 (예: 0.5)
    #
    # batched_nms():
    # - 클래스별로 NMS를 독립적으로 적용합니다.
    # - 같은 클래스에서 IoU가 임계값보다 크면 낮은 점수의 박스를 제거합니다.
    # - 서로 다른 클래스의 박스는 겹치더라도 서로 억제하지 않습니다.
    # - 제거되지 않은 박스들의 인덱스를 점수 내림차순으로 반환합니다.
    #
    # ids[...]:
    # - batched_nms()가 반환한 인덱스는 boxes[ids] 내부의 상대 인덱스입니다.
    # - 이를 원래 boxes 배열의 인덱스로 다시 변환합니다.
    #
    # 최종 keep: NMS 이후 살아남은 Bounding Box의 원본 인덱스
    keep = ids[
        batched_nms(
            boxes[ids],
            scores[ids],
            classes[ids],
            nms_iou
        )
    ]
    # <<< A02
    used=set();tp=0
    for i in keep.tolist():
        ovs=box_iou(boxes[[i]],gb)[0]
        eligible=[j for j in range(len(g)) if int(classes[i])==int(g.iloc[j].class_id) and j not in used]
        for j in sorted(eligible,key=lambda j:-float(ovs[j])):
            if ovs[j]>=0.5:used.add(j);tp+=1;break
    n=len(keep);fp=n-tp;fn=len(g)-tp
    rows.append(dict(conf=conf,nms_iou=nms_iou,kept=n,tp=tp,fp=fp,fn=fn,precision=tp/n if n else 0,recall=tp/len(g)))
r=pd.DataFrame(rows);r.to_csv(out/"nms_comparison.csv",index=False);print(r.round(4).to_string(index=False))
