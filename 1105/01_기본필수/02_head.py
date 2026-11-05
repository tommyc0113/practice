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
OUT = BASE

# 여기서는 [0,1] RGB 32×32로 학습한 교육용 source 가중치를 사용합니다.
# ImageNet ResNet18과 입력 변환을 섞지 않습니다.
# 이미지를 32×32 크기로 조정하고 Tensor로 변환
# 픽셀값을 0~1 범위로 정규화
transform = tr.Compose([
    tr.Resize((32, 32)),
    tr.ToTensor()
])

# 제공 부분: 0~8은 Backbone, 9는 새 3클래스 Head입니다.
model = nn.Sequential(
     # 첫 번째 합성곱 블록
    nn.Conv2d(3, 8, 3, padding=1, bias=False),  # RGB 3채널 → 특징 8개
    nn.BatchNorm2d(8),                         # 특징값 정규화
    nn.ReLU(),                                 # 음수값을 0으로 변경
    nn.MaxPool2d(2),                           # 이미지 크기를 절반으로 축소

    # 두 번째 합성곱 블록
    nn.Conv2d(8, 16, 3, padding=1, bias=False), # 특징 8개 → 16개
    nn.BatchNorm2d(16),                         # 특징값 정규화
    nn.ReLU(),                                  # 활성화 함수 적용

    # 특징 크기 조정
    nn.AdaptiveAvgPool2d((4, 4)),  # 특징 맵을 4×4 크기로 조정
    nn.Flatten(),                  # 2차원 특징을 1차원으로 펼침

    # 최종 분류기
    nn.Linear(256, 3)              # 특징 256개를 3개 클래스 점수로 변환
)

# 안전한 수치 배열만 읽습니다. Target 이미지로 사전학습한 파일이 아닙니다.
with np.load(DATA / "source_backbone.npz", allow_pickle=False) as saved:
    # 현재 모델의 가중치와 편향 정보를 딕셔너리로 가져오기
    state = model.state_dict()

    # npz 파일의 배열을 PyTorch Tensor로 변환하여 업데이트
    # copy()를 사용하여 배열 데이터를 복사
    state.update({
        name: torch.from_numpy(saved[name].copy())
        for name in saved.files
        })

model.load_state_dict(state)

for p in model.parameters():
    p.requires_grad = False
# TODO B02: 256개 특징에서 3개 logit을 출력하는 새 Linear Head를 만드세요.
# BEGIN B02
model[9] = nn.Linear(256, 3)  # 출력 개수를 고치세요.
# END B02
print("source 종류: 별도 합성 이미지로 사전학습한 TinyShapeCNN")
print("학습 파라미터:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("출력 shape:", tuple(model.eval()(torch.zeros(2, 3, 32, 32)).shape))
# PPT ResNet18의 구조도 다운로드 없이 확인합니다. 이것은 '전이학습 성능'이 아닙니다.
from torchvision.models import resnet18
resnet = resnet18(weights=None)
for p in resnet.parameters():
    p.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 3)
print("ResNet18 구조 확인용 Head 파라미터:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))
print("주의: 위 ResNet18은 구조만 확인했습니다. ImageNet 학습 결과가 아닙니다.")
