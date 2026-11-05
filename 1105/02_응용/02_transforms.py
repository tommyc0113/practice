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
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
transform = tr.Compose([tr.Resize((32, 32)), tr.ToTensor()])
from torchvision.models import ResNet18_Weights
from PIL import Image
weights = ResNet18_Weights.IMAGENET1K_V1  # Enum·변환 생성은 다운로드하지 않음
# TODO A02: Weight가 제공하는 transforms()를 가져오세요.
# BEGIN A02
official = weights.transforms() # weights.transforms()로 바꾸세요.
# END A02
image = Image.open(sorted((DATA/"valid"/"circle").glob("*.png"))[0]).convert("RGB")
print("소형 source 입력:",tuple(transform(image).shape),"범위 [0,1]")
print("공식 ResNet 입력:",tuple(official(image).shape))
print("공식 Resize/Crop:", official.resize_size,official.crop_size)
print("공식 mean/std:", official.mean,official.std)
print("작은 source와 ImageNet의 Normalize를 서로 섞지 않습니다.")
"""
20261105, task01
[결론]

교육용 TinyShapeCNN과 ImageNet 사전학습 ResNet18은 입력 이미지 크기와 정규화 기준이 서로 다르다는 것을 확인하였다.

TinyShapeCNN은 32×32 크기의 [0,1] RGB 이미지를 사용하지만, 
ResNet18은 224×224 크기와 ImageNet의 평균(mean)·표준편차(std)를 이용한 정규화가 필요하다.

따라서 사전학습 모델을 활용할 때는 해당 모델의 학습 조건에 맞는 전처리를 적용해야 하며, 
서로 다른 모델의 전처리 방식을 혼용하면 예측 성능이 저하될 수 있다.
"""