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

from PIL import Image, ImageDraw

# horizontal 폴더에서 PNG 파일을 이름순으로 정렬한 뒤 첫 번째 파일 선택
path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]

# 선택한 이미지를 열고 RGB 3채널 이미지로 변환
image = Image.open(path).convert("RGB")

# 학습 데이터에 적용할 데이터 증강 및 전처리 설정
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 이미지 크기를 64×64로 변경
    transforms.RandomHorizontalFlip(),  # 이미지를 무작위로 좌우 반전
    transforms.RandomRotation(10),  # 이미지를 -10도~10도 사이에서 무작위 회전
    transforms.ColorJitter(brightness=0.15),  # 밝기를 무작위로 변경
    transforms.ToTensor(),  # 이미지를 PyTorch Tensor로 변환하고 픽셀 값을 0~1로 정규화
])

# 좌우반전은 가로·세로 줄무늬 라벨을 보존한다. 90도 회전은 라벨을 바꾸므로 사용하지 않는다.
train_transform = transforms.Compose([
    transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15), transforms.ToTensor(),
])
# BEGIN A04
# TODO A04: valid에는 Resize와 ToTensor만 적용한다.
valid_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 검증 이미지 크기를 64×64로 변경
    transforms.ToTensor()  # 검증 이미지를 Tensor로 변환
])
# END A04
augmented = [train_transform(image) for _ in range(3)] # 학습 데이터 증강을 3번 실행하여 서로 다른 이미지 생성
v1, v2 = valid_transform(image), valid_transform(image) # 검증 이미지에 전처리를 2번 적용
canvas = Image.new("RGB", (4*128, 150), "white")
draw = ImageDraw.Draw(canvas) # 이미지 위에 글자를 표시하기 위한 객체 생성
for i, tensor in enumerate([v1]+augmented):
    canvas.paste(transforms.ToPILImage()(tensor).resize((128,128)), (128*i,22))
    draw.text((128*i+5,4), "valid" if i==0 else f"train {i}", fill="black")
canvas.save(out / "augmentation_preview.png")
print("valid 동일:", torch.equal(v1, v2))
print("shape:", tuple(v1.shape))
print("train 3개 동일 여부:", torch.equal(augmented[0], augmented[1]))
