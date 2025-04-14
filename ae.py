import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义自编码器模型
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # 如果输入是归一化的，使用Sigmoid
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train(model, dataloader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        total_loss = 0
        for batch_data, _ in dataloader:
            # 前向传播
            encoded_data, decoded_data = model(batch_data)
            loss = criterion(decoded_data, batch_data)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 打印每个epoch的损失
        if (epoch + 1) % 4 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}')


if __name__ == '__main__':
    # 生成示例数据
    input_dim = 784  # 例如MNIST数据28x28=784
    encoding_dim = 32  # 压缩后的维度
    X = torch.randn(1000, input_dim)  # 生成1000个样本
    # 训练模型
    num_epochs = 64
    learning_rate = 0.001

    # 数据加载器
    dataset = torch.utils.data.TensorDataset(X, X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型
    model = AutoEncoder(input_dim, encoding_dim)
    train(model, dataloader, num_epochs, learning_rate)