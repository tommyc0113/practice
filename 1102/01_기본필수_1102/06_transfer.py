"""실습 06 · 실제 ImageNet 사전학습 가중치로 Head만 학습한다.
최초 1회 공식 가중치 다운로드/캐시가 필요하다. 인터넷 없으면 교사에게 요청한다.
weights=None으로 바꾼 실험은 전이학습 결과가 아니다. 자동 대체하지 않는다.
"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights

torch.manual_seed(42)
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
weights = ResNet18_Weights.IMAGENET1K_V1 # ImageNet 데이터셋으로 사전 학습된 ResNet18 가중치를 선택합니다.
transform = weights.transforms()  # 이 가중치의 Resize·Crop·Normalize를 그대로 사용한다.
train = datasets.ImageFolder(root / "data" / "train", transform=transform)
valid = datasets.ImageFolder(root / "data" / "valid", transform=transform)
loader = DataLoader(train, batch_size=4, shuffle=False, num_workers=0)
model = resnet18(weights=weights)
# 해설 B06: 모든 기존 가중치의 미분을 끄고 새 Head를 붙인다.
# BEGIN B06
# TODO B06: 기존 parameter의 requires_grad를 False로 설정한다.
for parameter in model.parameters():
    parameter.requires_grad = False

# 기존 ResNet18의 마지막 분류기를 새로운 분류기로 교체합니다.
# in_features는 기존 ResNet18이 마지막에 전달하는 특징 개수입니다.
# 출력값 2는 새로 분류할 클래스가 2개라는 의미입니다.
model.fc = nn.Linear(model.fc.in_features, 2)

# 새로 만든 분류기 Head의 파라미터만 학습하도록 지정합니다.
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.0)

# 2개 클래스를 분류하기 위한 교차 엔트로피 손실 함수
loss_fn = nn.CrossEntropyLoss()

# eval은 역전파를 끄지 않는다. Backbone의 BN 통계까지 고정한다.
# Backbone의 BatchNorm 통계가 변경되지 않도록 평가 모드로 설정합니다.
model.eval()
model.fc.train()
for epoch in range(3):
    for images, labels in loader:
 
        optimizer.zero_grad()  # 이전 기울기를 초기화합니다.

        # 이미지 특징을 추출하고 Head에서 클래스 점수를 계산합니다.
        # 실제 정답과 비교해 손실값을 계산합니다.
        loss = loss_fn(model(images), labels)

        loss.backward()  # Head의 기울기를 계산합니다.
        optimizer.step() # Head의 가중치를 업데이트합니다.

model.eval()
with torch.no_grad():
    # 검증 데이터를 한 번에 불러옵니다.
    images, labels = next(
        iter(DataLoader(valid, batch_size=len(valid)))
    )

    # 모델의 예측 결과와 실제 정답을 비교해 정확도를 계산합니다.
    accuracy = (
        model(images).argmax(1) == labels
    ).float().mean().item()
    
print("실제 사전학습:", weights.name)
print("학습 파라미터:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("입력 크기:", tuple(images.shape))
print("valid 정확도:", round(accuracy, 4))
print("SmallCNN은 32×32 / ResNet은 224×224이므로 조건 차이를 기록한다.")
