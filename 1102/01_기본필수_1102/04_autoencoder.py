import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "series.csv")
# 시간 구간을 먼저 분리합니다. test 구간은 모델에 넣지 않습니다.
train = torch.tensor(
    df.loc[df["split"] == "train", "value"].values,
    dtype=torch.float32
)  # train 구간의 value 데이터를 추출하여 실수형 텐서로 변환합니다.

valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)  # valid 구간의 value 데이터를 추출하여 실수형 텐서로 변환합니다.

x = train.unfold(0, 8, 1).contiguous()  # 학습 데이터를 길이 8인 시계열 Window로 나누고 메모리를 연속적으로 정리합니다.
v = valid.unfold(0, 8, 1).contiguous()  # 검증 데이터도 길이 8인 Window로 나눕니다.

model = nn.Sequential(
    nn.Linear(8, 2),  # 입력 데이터 8개를 잠재 공간의 특징 2개로 압축합니다.
    nn.Tanh(),  # 압축된 값에 비선형 활성화 함수를 적용합니다.
    nn.Linear(2, 8)  # 잠재 특징 2개를 다시 원래 크기인 8개로 복원합니다.
)
optimizer = torch.optim.Adam(model.parameters(),lr=0.03) # Adam 옵티마이저를 사용하고 학습률을 0.03으로 설정합니다.
for epoch in range(40):
    optimizer.zero_grad()  # 이전 학습에서 계산한 기울기를 초기화합니다.

    reconstructed = model(x)  # 입력 데이터를 모델에 전달하여 압축한 뒤 원래 크기로 복원합니다.

    # TODO B04: Autoencoder의 정답을 입력 x 자체로 설정한다.
    # 정상 데이터를 0으로 바꾸는 과제가 아니다. reconstruction과 x를 비교한다.
    # BEGIN B04
    loss = nn.MSELoss()(reconstructed, x)  # 복원된 데이터와 원본 입력 데이터의 차이를 MSE로 계산합니다.
    loss.backward()  # 손실을 기준으로 각 모델 파라미터의 기울기를 계산합니다.

    optimizer.step()  # 계산된 기울기를 사용하여 모델의 가중치를 업데이트합니다.

model.eval()  # 모델을 평가 모드로 전환합니다.

with torch.inference_mode():  # 검증 단계에서는 기울기를 계산하지 않도록 설정합니다.

    latent = model[:2](v)  # Linear(8,2)와 Tanh까지만 실행하여 압축된 잠재 표현을 가져옵니다.

    error = nn.MSELoss()(model(v), v).item()  # 검증 데이터를 복원한 결과와 원본 데이터의 MSE를 계산합니다.
print("train:", tuple(x.shape), "valid:", tuple(v.shape))
print("latent / 복원:", tuple(latent.shape), tuple(model(v).shape))
print("valid MSE:", round(error,6))
