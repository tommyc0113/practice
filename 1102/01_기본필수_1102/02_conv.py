"""실습 02 · 공간 크기를 유지하고 특징 채널을 8개 만든다."""
import torch
from torch import nn

torch.manual_seed(42)
x = torch.rand(4, 3, 32, 32)
# Conv2d의 필수: 입력 채널, 출력 채널, 커널 크기.
# padding=1은 선택 인자. 3×3, stride=1과 결합하면 높이·너비를 유지한다.

# Conv2d의 필수: 입력 채널, 출력 채널, 커널 크기.
# padding=1은 선택 인자. 3×3, stride=1과 결합하면 높이·너비를 유지한다.
# BEGIN B02
# 2차원 합성곱 계층(Convolution Layer)을 생성합니다.
# 3         : 입력 채널 수(in_channels), RGB 이미지이므로 3개 채널
# 8         : 출력 채널 수(out_channels), 8개의 필터를 사용하여 8개의 특징 맵(Feature Map) 생성
# 3         : 커널 크기(kernel_size), 3×3 크기의 필터를 사용
# padding=1 : 이미지의 상하좌우에 1픽셀씩 패딩을 추가하여 공간 크기 감소를 방지
# stride    : 기본값 1, 필터가 한 번에 1픽셀씩 이동
# 입력 형태 : (4, 3, 32, 32) → 배치 4개, 입력 채널 3개, 높이 32, 너비 32
# 출력 형태 : (4, 8, 32, 32) → 배치 4개, 출력 채널 8개, 높이 32, 너비 32

# BEGIN B02
# TODO B02: 3입력·8출력·3×3·padding 1의 합성곱 층을 만든다.
conv = nn.Conv2d(3, 8, 3, padding=1)
y = conv(x)
print("입력:", tuple(x.shape))
print("출력:", tuple(y.shape))
print("파라미터:", sum(p.numel() for p in conv.parameters()))
