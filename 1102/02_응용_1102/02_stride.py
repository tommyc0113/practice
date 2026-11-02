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

x = torch.rand(4, 3, 32, 32) # 이미지 4장, RGB 3채널, 높이 32, 너비 32 생성
rows = []

# stride 값을 1부터 5까지 변경하면서 반복 실험
for stride in [1, 2, 3, 4, 5]:
    torch.manual_seed(42)
    # BEGIN A02
    # TODO A02: 현재 반복의 stride를 Conv2d에 전달한다.
    # 입력 채널 3개, 출력 채널 8개, 커널 3×3
    # stride는 반복문에서 전달받은 값 사용
    # padding=1은 이미지의 가장자리에 패딩을 추가
    conv = nn.Conv2d(3, 8, 3, stride=stride, padding=1)
    
    y = conv(x) # 입력 이미지 x에 합성곱 연산 적용

    # 모델의 전체 파라미터 개수 계산
    # weight와 bias에 해당하는 원소 개수를 모두 합산
    params = sum(p.numel() for p in conv.parameters())
    
    # 1 MAC=곱셈 후 누적 1회. bias 덧셈은 제외하며 batch 전체 값이다.
    macs = y.numel() * (3 * 3 * 3)
    
    # 현재 stride의 결과를 딕셔너리 형태로 저장
    rows.append(dict(
        stride=stride,  # 현재 stride 값
        shape=str(tuple(y.shape)),  # 출력 Tensor의 크기
        params=params,  # 파라미터 개수
        macs=macs  # MAC 연산 횟수
    ))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "stride_comparison.csv", index=False)
