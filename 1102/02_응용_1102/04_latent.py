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
# train 데이터의 value 열을 PyTorch 텐서로 변환
train = torch.tensor(
    df.loc[df["split"] == "train", "value"].values,
    dtype=torch.float32
)

# valid 데이터의 value 열을 PyTorch 텐서로 변환
valid = torch.tensor(
    df.loc[df["split"] == "valid", "value"].values,
    dtype=torch.float32
)

# 시계열 데이터에서 연속된 8개 값을 하나의 학습 샘플로 구성
# unfold(0, 8, 1): 0번 차원에서 크기 8의 윈도우를 1칸씩 이동
# contiguous(): 텐서 데이터를 연속된 메모리 형태로 정리
# 결과 Shape: (생성된 윈도우 수, 8)
# 길이 8, 간격 1인 슬라이딩 윈도우 생성
x = train.unfold(0, 8, 1).contiguous()

# 검증 데이터도 동일하게 길이 8의 윈도우 생성
v = valid.unfold(0, 8, 1).contiguous()

rows = []  # 잠재 공간 크기별 결과를 저장할 리스트

for latent_size in [1,2,4]:

    torch.manual_seed(42)
    # TODO A04: 후보 latent_size=1·2·4를 실제 Encoder·Decoder 크기에 적용한다.
    # BEGIN A04

# Autoencoder 모델 구성
# Encoder : Linear(8, 2) → 입력 특성 8개를 2개로 압축
# Tanh    : -1~1 범위의 비선형 활성화 함수
# Latent  : 압축된 2차원 특징 표현
# Decoder : Linear(2, 8) → 압축된 특징 2개를 8개로 복원
    model = nn.Sequential(
    nn.Linear(8, latent_size),  #
    nn.Tanh(),                  # 
    nn.Linear(latent_size, 8)    # 
    )
    
    # Adam 옵티마이저 생성
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    best = float("inf")  # 가장 낮은 검증 손실을 저장할 변수    
    
    for epoch in range(1,41):
        optimizer.zero_grad()          # 이전 기울기 초기화
        loss = nn.MSELoss()(model(x), x)  # 입력 데이터와 복원 데이터의 차이 계산
        loss.backward()                # 역전파로 기울기 계산
        optimizer.step()               # 가중치 업데이트

        # 검증 단계에서는 기울기를 계산하지 않음
        with torch.inference_mode():
            score = nn.MSELoss()(model(v), v).item()  # 검증 MSE 계산

        # 지금까지 가장 낮은 검증 손실인지 확인
        if score < best:
            best = score       # 최저 검증 손실 갱신
            best_epoch = epoch # 최저 손실이 나온 학습 횟수 저장

    # 모델의 학습 결과를 리스트에 추가
    rows.append({
        "latent": latent_size,                  # 잠재 공간 크기
        "retained_ratio": latent_size / 8,      # 압축 후 유지되는 차원 비율
        "params": sum(p.numel() for p in model.parameters()),  # 전체 파라미터 개수
        "best_epoch": best_epoch,               # 최저 검증 손실이 나온 epoch
        "valid_mse": best                       # 최저 검증 MSE
    })

table=pd.DataFrame(rows); table.to_csv(OUT/"latent_comparison.csv",index=False)
print(table.round(6).to_string(index=False))
