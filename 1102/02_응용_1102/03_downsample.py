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

x = torch.rand(4, 8, 32, 32) # 이미지 4장, 채널 8개, 높이 32, 너비 32 생성
operations = {
    "pool": nn.MaxPool2d(2),  # 2×2 최대 풀링으로 공간 크기를 절반으로 축소
    "conv_stride2": nn.Conv2d(8, 8, 3, stride=2, padding=1)  # stride=2인 합성곱으로 공간 크기 축소
}
rows = []

# 딕셔너리에서 이름과 신경망 층을 하나씩 가져와 반복
for name, layer in operations.items():
    # BEGIN A03
    # TODO A03: 각 downsampling 층을 입력 x에 적용한다.
    y = layer(x)
    
# 연산 이름, 출력 크기, 파라미터 개수를 딕셔너리로 저장
    rows.append(dict(
        method=name,  # Downsampling 방법 이름
        shape=str(tuple(y.shape)),  # 연산 결과 Tensor의 크기
        params=sum(p.numel() for p in layer.parameters())  # 학습 가능한 파라미터 개수
    ))

print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(out / "downsample_comparison.csv", index=False)
