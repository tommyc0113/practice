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

torch.set_num_threads(1)
torch.manual_seed(14)
BASE = Path(__file__).resolve().parent  # 현재 파이썬 파일이 위치한 폴더 경로
DATA = BASE / "data"                    # 데이터 폴더 경로
OUT = BASE                              # 결과물을 저장할 폴더 경로

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])

# TODO B01: ImageFolder로 train 폴더를 읽으세요. root는 필수 경로입니다.
# BEGIN B01
train = ImageFolder(DATA / "train")
# END B01
print("class_to_idx:", train.class_to_idx)
for split in ["train", "valid", "test"]:
    # 해당 폴더의 이미지를 ImageFolder로 읽기
    dataset = ImageFolder(DATA / split, transform=transform)

    # 각 클래스(0, 1, 2)에 해당하는 이미지 개수 계산
    # dataset.targets에는 이미지마다 부여된 정답 레이블이 저장됨
    counts = [dataset.targets.count(i) for i in range(3)]

    print(split, "개수:", counts, "같은 매핑:", dataset.class_to_idx == train.class_to_idx)
# test는 파일명·라벨 매핑만 확인했습니다. 모델 예측에는 사용하지 않습니다.
