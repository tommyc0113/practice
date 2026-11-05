"""Day14: 제공된 정상 이미지로 학습합니다. test는 읽지 않습니다."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms as tr
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

torch.set_num_threads(1)  # CPU 연산에 사용할 스레드 수를 1개로 제한
torch.manual_seed(14)    # 난수 시드를 14로 고정하여 재현성 확보

BASE = Path(__file__).resolve().parent  # 현재 파이썬 파일의 위치
DATA = BASE / "data"                    # 원본 이미지 데이터 폴더
OUT = BASE / "outputs"                  # 결과 저장 폴더
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.

# 이미지를 32×32 크기로 변경하고 Tensor로 변환
# ToTensor()는 픽셀값을 0~1 범위로 정규화
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
from collections import defaultdict
import hashlib
rows = []
seen = defaultdict(list)
for split in ["train", "valid", "test"]:
    dataset = ImageFolder(DATA / split)
    for label, name in enumerate(dataset.classes):
        # TODO A01: 해당 라벨의 개수를 세세요.
        # BEGIN A01
        count = dataset.targets.count(label)  # dataset.targets에서 label의 수를 세세요.
        # END A01

        # 데이터셋 이름, 클래스 이름, 이미지 개수, 비율 저장
        rows.append({"split": split, "class": name, "n": count, "ratio": count/len(dataset)})
    for path,_ in dataset.samples:
        seen[hashlib.sha256(Path(path).read_bytes()).hexdigest()].append(str(Path(path).relative_to(DATA)))
table = pd.DataFrame(rows)
table.to_csv(OUT/"class_counts.csv", index=False)
print(table.round(4).to_string(index=False))
duplicates = [paths for paths in seen.values() if len(paths)>1]
print("바이트 동일 중복 그룹:",len(duplicates))
(OUT/"duplicates.json").write_text(json.dumps(duplicates,ensure_ascii=False,indent=2),encoding="utf-8")
# test 픽셀은 중복 감사에만 쓰며 모델 성능을 계산하지 않습니다.
#   누수가능성 설명
#   바이트 단위의 동일 중복 이미지는 발견되지 않았으나, 
#   Train·Valid·Test 간 시각적으로 유사한 이미지나 동일 원본에서 변형된 이미지가 포함되어 있을 경우 
#   데이터 누수(Data Leakage)가 발생할 가능성이 있다.