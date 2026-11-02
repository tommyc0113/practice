import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"data"/"series.csv")

# 학습 데이터와 검증 데이터를 분리합니다.
train = torch.tensor(
    df.loc[df["split"] == "train", "value"].values,
    dtype=torch.float32
)  # train 구간의 value 열을 추출하여 실수형 텐서로 변환합니다.

valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)  # valid 구간의 value 열을 추출하여 실수형 텐서로 변환합니다.

from copy import deepcopy
w = train.unfold(0, 13, 1)  # 학습 데이터를 과거 12개와 정답 1개로 구성된 Window로 분리합니다.
v = valid.unfold(0, 13, 1)  # 검증 데이터도 동일한 방식으로 Window로 분리합니다.

x, y = w[:, :12].unsqueeze(-1), w[:, 12:13]  # 학습 입력은 과거 12개, 정답은 다음 값 1개로 구성합니다.
vx, vy = v[:, :12].unsqueeze(-1), v[:, 12:13]  # 검증 입력과 검증 정답도 동일한 방식으로 구성합니다.

rows=[]
for name in ["RNN","LSTM"]: # RNN과 LSTM 모델을 차례대로 비교합니다.

    torch.manual_seed(42)  # 각 모델의 초기 가중치 생성을 위한 난수 시드를 고정합니다.

    core = nn.RNN(1, 8, batch_first=True) if name == "RNN" else nn.LSTM(1, 8, batch_first=True)
    # RNN 또는 LSTM 모델을 생성합니다. 입력 특징은 1개이고 은닉 상태는 8개입니다.

    head = nn.Linear(8, 1)  # 은닉 특징 8개를 다음 시점의 예측값 1개로 변환합니다.

    params = list(core.parameters()) + list(head.parameters())  # RNN/LSTM과 Linear 층의 학습 파라미터를 하나로 합칩니다.

    optimizer = torch.optim.Adam(params, lr=0.02)  # Adam 옵티마이저를 생성하고 학습률을 0.02로 설정합니다.

    best = float("inf")  # 가장 좋은 검증 MSE를 저장할 변수를 무한대로 초기화합니다.

    for epoch in range(1,41):
        core.train()  # RNN 또는 LSTM을 학습 모드로 설정합니다.
        head.train()  # Linear 층을 학습 모드로 설정합니다.

        optimizer.zero_grad()  # 이전 반복에서 계산한 기울기를 초기화합니다.

        output, _ = core(x)  # 학습 데이터를 모델에 입력하고 모든 시점의 은닉 출력을 가져옵니다.
        
        # TODO A03: 각 과거 Window의 마지막 출력에서 다음 값에 대한 MSE를 계산한다.
        # BEGIN A03
        loss = nn.MSELoss()(head(output[:, -1, :]), y)       

        loss.backward()  # 손실을 기준으로 각 파라미터의 기울기를 계산합니다.

        optimizer.step()  # 계산된 기울기를 이용하여 모델의 가중치를 업데이트합니다.

        core.eval()  # 검증을 위해 RNN 또는 LSTM을 평가 모드로 전환합니다.
        head.eval()  # Linear 층을 평가 모드로 전환합니다.

        with torch.inference_mode():  # 검증 단계에서는 기울기를 계산하지 않도록 설정합니다.

            output, _ = core(vx)  # 검증 데이터를 모델에 입력하여 은닉 출력을 계산합니다.

            score = nn.MSELoss()(head(output[:, -1]), vy).item()
            # 마지막 시점의 예측값과 검증 정답의 MSE를 계산하고 파이썬 숫자로 변환합니다.

        if score < best:  # 현재 검증 MSE가 이전 최솟값보다 작으면 실행합니다.

            best = score  # 가장 낮은 검증 MSE를 저장합니다.

            best_epoch = epoch  # 가장 좋은 성능을 기록한 학습 횟수를 저장합니다.

    rows.append({
        "model": name,  # 모델 이름을 저장합니다.
        "hidden": 8,  # 은닉 상태의 크기를 저장합니다.
        "params": sum(p.numel() for p in params),  # 모델 전체 학습 파라미터 개수를 계산합니다.
        "epochs": 40,  # 전체 학습 횟수를 저장합니다.
        "best_epoch": best_epoch,  # 가장 좋은 검증 성능을 기록한 epoch를 저장합니다.
        "valid_mse": best  # 가장 낮았던 검증 MSE를 저장합니다.
    })

    # 마지막 관측값을 다음 값으로 사용하는 학습 없는 기준선도 기록합니다.
    rows.append({
        "model": "Persistence",  # 모델 이름을 Persistence로 지정합니다.
        "hidden": 0,  # 신경망의 은닉 상태가 없으므로 0으로 설정합니다.
        "params": 0,  # 학습 파라미터가 없으므로 0으로 설정합니다.
        "epochs": 0,  # 별도의 학습 과정이 없으므로 0으로 설정합니다.
        "best_epoch": 0,  # 학습을 하지 않으므로 0으로 설정합니다.
        "valid_mse": nn.MSELoss()(vx[:, -1], vy).item()  # 마지막 입력값을 다음 값의 예측값으로 사용하여 MSE를 계산합니다.
    })
table=pd.DataFrame(rows); table.to_csv(OUT/"forecast_comparison.csv",index=False)
print("windows:",len(x),len(vx),"test 미사용")
print(table.round(6).to_string(index=False))
