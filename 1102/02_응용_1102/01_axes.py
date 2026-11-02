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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# horizontal 폴더에서 PNG 파일 목록을 가져와 이름순으로 정렬한 후 첫 번째 파일 선택
path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]
# 이미지 파일을 열고 RGB로 변환한 뒤 PyTorch Tensor로 변환
# ToTensor()는 이미지의 픽셀 값을 0~1 범위로 정규화하고 CHW 형태로 변환
chw = transforms.ToTensor()(Image.open(path).convert("RGB"))

# BEGIN A01
# TODO A01: 채널 축을 마지막으로 보내 HWC를 만든다.
hwc = chw.permute(1, 2, 0)

# HWC를 다시 CHW로 변경한 다음 맨 앞에 배치 차원 N을 추가
# 결과 형태는 NCHW가 됨
nchw = hwc.permute(2, 0, 1).unsqueeze(0)

plt.figure(figsize=(4, 4))
plt.imshow(hwc.numpy())
plt.title("HWC view: horizontal stripes")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "image_axes.png", dpi=120)
plt.close()
print("CHW / HWC / NCHW:", tuple(chw.shape), tuple(hwc.shape), tuple(nchw.shape))
print("왕복 값 일치:", torch.equal(chw, nchw[0]))
