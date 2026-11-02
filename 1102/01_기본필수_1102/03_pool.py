"""실습 03 · 클래스 정의 없이 Sequential로 작은 CNN을 연결한다."""
import torch
from torch import nn

torch.manual_seed(42)
x = torch.rand(4, 3, 32, 32)  # 이미지 4장, RGB 3채널, 높이 32, 너비 32 생성
# MaxPool2d(2)는 2×2에서 최댓값을 선택한다. 기본 stride도 2이다.
# BEGIN B03
# TODO B03: 출력 높이·너비를 절반으로 만드는 MaxPool을 작성한다.
pool = nn.MaxPool2d(2)

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),  # 입력 채널 3개, 출력 채널 16개, 3×3 합성곱
    nn.ReLU(),  # 음수 값을 0으로 변경하는 활성화 함수
    pool,  # 2×2 최대 풀링으로 이미지 크기를 절반으로 축소

    nn.Conv2d(16, 32, 3, padding=1),  # 16개 채널을 입력받아 32개 특징 채널 생성
    nn.ReLU(), 
    nn.MaxPool2d(2),  # 다시 높이와 너비를 절반으로 축소

    nn.AdaptiveAvgPool2d(1),  # 각 특징 채널의 공간 크기를 1×1로 변환
    nn.Flatten(),  # 2차원 이상의 데이터를 1차원으로 펼침
    nn.Linear(32, 2),  # 32개 입력 특징을 2개 출력값으로 변환
)
for layer in model:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
