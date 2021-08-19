"""
옵션: Data Parallelism
==========================
**저자**: `Sung Kim <https://github.com/hunkim>`_ and `Jenny Kang <https://github.com/jennykang>`_

이 튜토리얼에서, 우리는 ``DataParallel`` 를 활용하여 멀티 GPU를 활용하는법을 배워볼 것입니다.

파이토치에서 GPU를 쓰는 법은 매우 쉽습니다. GPU에서 모델을 할당 할 수 있습니다.

.. code:: python

    device = torch.device("cuda:0")
    model.to(device)

다음, GPU에 모든 텐서를 복사할 수 있습니다.

.. code:: python

    mytensor = my_tensor.to(device)

``my_tensor`` 를 다시 쓰는 대신에, ``my_tensor.to(device)`` 를 선언하여 GPU에 ``my_tensor`` 를 되돌리는 것임을 주목해주시기 바랍니다.
새로운 텐서를, GPU 텐서로 할당하고 GPU 상에서 사용할 필요가 있습니다. 

멀티 GPU상에서 순전파 및 역전파를 실행하는 것은 당연합니다.
파이토치는 단일 GPU를 기본적으로 지원하지만, 멀티 GPU에서도 ``DataParallel`` 을 사용하여 병렬적으로 모델 학습을 수행할 수 있습니다.  

.. code:: python

    model = nn.DataParallel(model)

위에서 정의한 코드는, 이 튜토리얼의 핵심입니다. 아래에서 더욱 상세하게 설명하겠습니다.
"""


######################################################################
# 불러오기 및 파라미터
# ---------------
#
# 파이토치 모듈을 불러오고 파라미터를 정의합니다.
#

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


######################################################################
# 장치
# -----------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# 허위의 데이터셋
# -------------
#
# 허위의 데이터셋을 구성합니다. 단지, __getitem__ 함수를 구현할 때 필요합니다.
#

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)


######################################################################
# 간단한 모델
# ------------
#
# For the demo, our model just gets an input, performs a linear operation, and
# 데모에서, 우리는 입력 얻고, 선형 연산을 하고, 결과값을 주었습니다. 그러나 ``DataParallel``은 
# 어떠한 다양한 모델에서도(CNN, RNN, Capsule Net etc.) 활용할 수 있습니다.
#
# 우리는 모델 안에 있는 입력값의 사이즈와 출력값 텐서를 확인하기 위한 출력문을 작성했습니다. 
# batch rank 0에서 출력되다는 점에 주의를 부탁드립니다.
#

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output


######################################################################
# 모델 만들기 및 DataParallel
# -----------------------------
#
# 이 튜토리얼의 핵심입니다. 첫째로, 모델 인스턴스를 만들어야 하고, 멀티 GPU의 작동 여부를 확인합니다.
# 만약 멀티 GPU를 가지고 있다면, ``nn.DataParallel``을 활용하여 모델을 감싸줍니다. 
# 다음, ``model.to(device)`` 활용하여 모델을 GPU로 할당합니다.
#

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)


######################################################################
# 모델 학습
# -------------
#
# 여기서는 입력, 결과 텐서의 사이즈를 확인 할 수 있습니다. 
#

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())


######################################################################
# 결과
# -------
#
# 보유한 GPU가 없거나, 30개의 입력 및 결과를 하나의 배치로 설정할 때, 모델은 예상한 바와 같이, 30개의 추론과, 30개의 결과값을 갖게 됩니다.
# 그러나 만약 멀티 GPU를 보유한다면, 다음과 같은 결과를 낼 수 있습니다.  
#
# 2 GPUs
# ~~~~~~
#
# If you have 2, you will see:
#
# .. code:: bash
#
#     # on 2 GPUs
#     Let's use 2 GPUs!
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#         In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#         In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 3 GPUs
# ~~~~~~
#
# If you have 3 GPUs, you will see:
#
# .. code:: bash
#
#     Let's use 3 GPUs!
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#         In Model: input size torch.Size([10, 5]) output size torch.Size([10, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#
# 8 GPUs
# ~~~~~~~~~~~~~~
#
# If you have 8, you will see:
#
# .. code:: bash
#
#     Let's use 8 GPUs!
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([4, 5]) output size torch.Size([4, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#         In Model: input size torch.Size([2, 5]) output size torch.Size([2, 2])
#     Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
#


######################################################################
# 요약
# -------
#
# DataParallel은 자동으로 데이터를 나누고 여러개의 GPU에 작업 순서를 매겨 보냅니다.  
# 각 모델이, 작업을 마친 후, DataParallel은 값을 개발자에게 되돌려주기 전에, 결과를 모으고 수집합니다. 
#
# 더욱 자세한 정보를 원하면 아래의 사이트를 참고 부탁드립니다.
# https://pytorch.org/tutorials/beginner/former\_torchies/parallelism\_tutorial.html.
#