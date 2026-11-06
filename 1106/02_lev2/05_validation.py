"""실습 05. 실제 준비 모델을 valid에 평가합니다. test는 전달하지 않습니다."""
from pathlib import Path
import json,yaml,torch
import pandas as pd
from ultralytics import YOLO
BASE=Path(__file__).resolve().parent
if __name__=="__main__":
    torch.set_num_threads(1)
    out=BASE/"outputs";out.mkdir(exist_ok=True)
    config={"train":str(BASE/"data/images/train"),"val":str(BASE/"data/images/val"),"names":["square","circle","triangle"]}
    path=out/"dataset_active.yaml";path.write_text(yaml.safe_dump(config),encoding="utf-8")
    # model=YOLO(str(BASE/"data/demo.pt"))  # 실제 학습 후 best.pt를 사용하는것이 아니라면 샘플 demo.pt를 사용합니다. 아래 코드는 주석처리하셔야해요.
    model=YOLO(str(BASE/"outputs/smoke/weights/best.pt"))  # 실제 학습 후 best.pt를 사용합니다.
    # >>> A05
    # 요구: model.val에 data=path, split='val'을 전달하세요.
    # 해설: val의 iou 옵션과 mAP50-95에서 쓰는 여러 평가 IoU를 혼동하지 않습니다.
    metrics=model.val(data=str(path),split="val",imgsz=160,batch=4,workers=0,
                      device="cpu",plots=False,project=str(out),name="validation",exist_ok=True)
    # <<< A05
    summary={"map50":float(metrics.box.map50),"map50_95":float(metrics.box.map),"split":"val"}
    (out/"validation_metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    rows=[]
    for i,cls in enumerate(metrics.box.ap_class_index):
        pr,rc,ap50,ap=metrics.box.class_result(i)
        rows.append([int(cls),model.names[int(cls)],float(pr),float(rc),float(ap50),float(ap)])
    pd.DataFrame(rows,columns=["class_id","name","precision","recall","ap50","ap50_95"]).to_csv(out/"class_metrics.csv",index=False)
    print(summary)

"""
20261109,   TASK01

최종 결론: 
학습·검증 파이프라인의 실행은 확인되었으나, 실제 객체 탐지 성능은 활용 가능한 수준에 도달하지 못하였다. 
향후 사전학습 가중치 활용, Epoch 증가, 클래스 불균형 점검 및 입력 해상도 조정을 통해 성능을 개선할 필요가 있다.
"""