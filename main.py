import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model import AutoEncoder, ae_train

# 定义超参数

AE_INPUT_DIM = None
AE_ENCODE_DIM = 32
AE_BATCH_SIZE = 128
AE_EPOCHS = 64
AE_LEARNING_RATE = 0.01

# 数据导入

input_data = pd.read_csv()

# 对象实例化

model = AutoEncoder(input_dim=AE_INPUT_DIM, encoding_dim=AE_ENCODE_DIM)
ae_dataloader = DataLoader(dataset=input_data, batch_size=AE_BATCH_SIZE, shuffle=True)

# 训练降维模型

ae_train(model=model, dataloader=ae_dataloader, num_epochs=AE_EPOCHS, learning_rate=AE_LEARNING_RATE)

# 因子回测

# 训练预测模型

# 策略回测