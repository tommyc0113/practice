import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)

from torch.nn.utils.rnn import pad_sequence
sequences=[torch.ones(4,3),torch.ones(2,3)*2,torch.ones(5,3)*3]
lengths=torch.tensor([4,2,5]); padded=pad_sequence(sequences,batch_first=True)
mask=torch.arange(padded.size(1)).unsqueeze(0)>=lengths.unsqueeze(1)
# 해설 A01: Padding이 아닌 실제 시점만 합산하고 각 자료의 실제 길이로 나눈다.
# mask=True는 Padding이다. 0도 정상 관측값일 수 있으므로 값이 0인지로 판정하지 않는다.
# BEGIN A01
# 패딩(Padding)된 부분을 제외하고, 각 시퀀스의 실제 데이터에 대한 평균 벡터를 계산합니다.
#
# padded                  : 패딩이 포함된 입력 텐서 (Batch, Seq_len, Features)
# mask                    : 패딩 위치를 표시하는 Boolean 텐서 (True=패딩, False=실제 데이터)
# ~mask                   : mask를 반전하여 실제 데이터 위치만 True로 변경
# unsqueeze(-1)           : 마지막 차원을 추가하여 (Batch, Seq_len, 1) 형태로 변환
# padded * (~mask)...     : 패딩 위치의 값을 0으로 만들어 평균 계산에서 제외
# sum(dim=1)              : 시퀀스 길이 방향으로 실제 데이터의 값들을 합산
# lengths.unsqueeze(-1)   : 각 시퀀스의 실제 길이를 (Batch, 1) 형태로 변환
# / lengths...            : 실제 데이터 개수로 나누어 평균 벡터 계산
#
# 입력 Shape  : padded = (Batch, Seq_len, Features)
# 출력 Shape  : mean   = (Batch, Features)
mean = (padded * (~mask).unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
# 값이 복잡해 보이죠? 실제로는 패딩된 부분을 제외하고 각 시퀀스의 평균을 계산하는 간단한 연산입니다.
#   mean = sum(scores) / len(scores) 이게 원래 식이에요.
#   그런데 패딩값을 빼야해서 (~mask)로 실제 값만 남기고, sum(dim=1)으로 합산하고, lengths로 나누어 평균을 구하는 거죠.
# END A01
print("Padding mask:\n",mask)
print("실제 값 평균:\n",mean)
pd.DataFrame(mask.int().numpy()).to_csv(OUT/"padding_mask.csv",index=False)