import torch
from torch import nn
torch.set_num_threads(1)  # 작은 CPU 실습의 실행 조건을 고정한다.
torch.manual_seed(42)
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
OUT.mkdir(exist_ok=True)

x=torch.randn(4,6,3); core=nn.RNN(3,8,batch_first=True); head=nn.Linear(8,2)

# RNN 순전파
# output : 모든 시점의 은닉 상태, Shape (4, 6, 8)
# h      : 마지막 시점의 최종 은닉 상태, Shape (1, 4, 8)
output, h = core(x)

# Many-to-One: 마지막 시점의 은닉 상태만 분류
# output[:, -1] : 모든 배치에서 마지막 시점 선택 → (4, 8)
# head 적용 후 출력 Shape : (4, 2)
many_to_one = head(output[:, -1])

# 해설 A02: 모든 시점의 hidden을 같은 Linear 층에 넣어 시점별 logits를 만든다.
# Linear는 마지막 차원에 적용되므로 [B,T,H]→[B,T,C]로 작동한다.
# 입력 Shape  : (4, 6, 8)
# 출력 Shape  : (4, 6, 2)
# BEGIN A02
many_to_many=head(output)
# END A02
print("one:",tuple(many_to_one.shape),"many:",tuple(many_to_many.shape))
print("마지막 logits 일치:",torch.allclose(many_to_many[:,-1],many_to_one))
print("같은 Head를 시점마다 재사용하며, 시점별 정답은 별도 설계가 필요하다.")