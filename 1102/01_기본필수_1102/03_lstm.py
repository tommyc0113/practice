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
)  # split 열이 train인 행에서 value 값을 추출해 실수형 텐서로 변환합니다.

valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)  # split 열이 valid인 행에서 value 값을 추출해 실수형 텐서로 변환합니다.

# unfold(0,13,1): 과거 12개+다음 정답 1개인 묶음을 만든다.
window = train.unfold(0,13,1)
x = window[:, :12].unsqueeze(-1)  # 각 구간의 앞 12개 값을 입력 데이터로 선택하고 마지막 차원을 추가합니다.
y = window[:, 12:13]  # 각 구간의 마지막 1개 값을 예측해야 할 정답으로 선택합니다.

lstm = nn.LSTM(1, 8, batch_first=True)  # 입력 특징 1개와 은닉 상태 8개를 사용하는 LSTM 모델을 생성합니다.
head = nn.Linear(8, 1)  # LSTM의 은닉 특징 8개를 다음 시점의 예측값 1개로 변환하는 층을 만듭니다.

output, (h_n, c_n) = lstm(x)  # 입력 데이터를 LSTM에 전달하고 모든 시점의 출력, 마지막 은닉 상태와 Cell State를 받습니다.

# TODO B03: 12개 관측의 마지막 은닉 출력으로 다음 값 1개를 예측한다.
# [:, -1, :]는 마지막 시점이다. 이 예제의 Window 길이는 모두 12이다.
# BEGIN B03
prediction = head(output[:, -1, :])
print("입력 / 정답:", tuple(x.shape), tuple(y.shape))
print("h_n / c_n:", tuple(h_n.shape), tuple(c_n.shape))
print("예측:", tuple(prediction.shape))
print("첫 예측:", round(prediction[0].item(),6))
print("학습 전 MSE:", round(nn.MSELoss()(prediction,y).item(),6))
