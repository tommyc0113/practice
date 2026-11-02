from pathlib import Path
import copy
import csv
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

torch.manual_seed(42)
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 모든 이미지를 32x32 크기로 변경합니다.
    transforms.ToTensor()          # 이미지를 PyTorch 텐서로 변환합니다.
])

# ImageFolder를 사용해 이미지와 클래스 라벨을 불러옵니다.
# 폴더 이름이 클래스 이름이 되고, 각 폴더의 이미지가 해당 클래스로 분류됩니다.
train = datasets.ImageFolder(root / "data" / "train", transform=transform)
valid = datasets.ImageFolder(root / "data" / "valid", transform=transform)

train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(42))
valid_loader = DataLoader(valid, batch_size=16, shuffle=False, num_workers=0)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),  # 입력 채널 3개(RGB)를 16개 특징 맵으로 변환합니다.
    nn.ReLU(),                       # 음수 값을 0으로 만들어 비선형성을 추가합니다.
    nn.MaxPool2d(2),                 # 2x2 영역에서 최댓값을 선택해 이미지 크기를 절반으로 줄입니다.

    nn.Conv2d(16, 32, 3, padding=1), # 16개 특징 맵을 32개 특징 맵으로 변환합니다.
    nn.ReLU(),                       # 활성화 함수 ReLU를 적용합니다.
    nn.MaxPool2d(2),                 # 특징 맵의 가로와 세로 크기를 다시 절반으로 줄입니다.

    nn.AdaptiveAvgPool2d(1),         # 각 특징 맵을 1x1 크기로 평균 풀링합니다.
    nn.Flatten(),                    # 2차원 특징을 1차원 벡터로 펼칩니다.
    nn.Linear(32, 2),                # 32개 특징을 2개 클래스의 점수로 변환합니다.
)

loss_fn = nn.CrossEntropyLoss()  # 다중 클래스 분류에 사용하는 교차 엔트로피 손실 함수
optimizer = torch.optim.AdamW(
    model.parameters(),  # 모델의 학습 가능한 가중치와 편향을 전달합니다.
    lr=0.01,              # 학습률: 가중치를 얼마나 크게 수정할지 결정합니다.
    weight_decay=0.0      # 가중치 감소를 적용하지 않습니다.
)
history = []
best_loss = float("inf")
for epoch in range(1, 6):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        # BEGIN A05
        # TODO A05: loss 역전파와 가중치 갱신을 순서대로 실행한다.
        loss.backward()  # 손실값을 기준으로 각 파라미터의 기울기를 계산합니다.
        optimizer.step() # 계산된 기울기를 사용해 모델의 가중치와 편향을 업데이트합니다.
    # 같은 epoch 말 모델을 eval 모드로 평가해 train/valid의 측정 조건을 맞춘다.
    model.eval()
    row = {"epoch": epoch}
    
    with torch.no_grad():
        for phase, loader in [("train", DataLoader(train, batch_size=16)), ("valid", valid_loader)]:
            total_loss, correct, count = 0.0, 0, 0
            for images, labels in loader:
                logits = model(images)  # 모델의 예측 점수를 계산합니다.

                # 배치의 평균 손실에 이미지 개수를 곱해 전체 손실 합을 계산합니다.
                total_loss += loss_fn(logits, labels).item() * len(labels)

                # 가장 높은 점수를 받은 클래스 번호를 예측 결과로 선택합니다.
                # 예측값과 실제 정답이 같으면 정답 개수를 증가시킵니다.
                correct += (logits.argmax(1) == labels).sum().item()

                count += len(labels)
            row[phase+"_loss"] = total_loss/count
            row[phase+"_accuracy"] = correct/count
    history.append(row)
    if row["valid_loss"] < best_loss:
        best_loss, best_epoch = row["valid_loss"], epoch
        best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
model.eval()
actual, predicted = [], []

# 검증 데이터로 최적 모델의 최종 예측을 수행합니다.
with torch.no_grad():

    for images, labels in valid_loader:

        actual.extend(labels.tolist())  # 실제 정답을 리스트에 추가합니다.

        # 모델의 예측 점수 중 가장 높은 클래스 번호를 저장합니다.
        predicted.extend(model(images).argmax(1).tolist())

# 2개 클래스를 분류하는 혼동행렬을 0으로 초기화합니다.
cm = torch.zeros(2, 2, dtype=torch.long)

for truth, guess in zip(actual, predicted):
    cm[truth, guess] += 1
print(pd.DataFrame(history).round(4).to_string(index=False))
print("best epoch:", best_epoch)
print("혼동행렬 행=실제, 열=예측:\n", cm.tolist())
print("클래스별 정답 수:", cm.diag().tolist(), "/", cm.sum(1).tolist())
pd.DataFrame(history).to_csv(out / "cnn_history.csv", index=False)
pd.DataFrame(cm.numpy(), index=train.classes, columns=train.classes).to_csv(out / "confusion_matrix.csv")
pd.DataFrame(dict(file=[Path(p).name for p,_ in valid.samples], actual=actual, predicted=predicted)).to_csv(out / "cnn_predictions.csv", index=False)
