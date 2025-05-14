import torch
import torch.nn as nn

class SquareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 100),    
            nn.ReLU(),           
            nn.Linear(100, 1)     
        )
    
    def forward(self, x):
        return self.fc(x)

model = SquareNet()
criterion = nn.MSELoss()                # 均方误差损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 自适应学习率

# 转换为PyTorch张量
x = range(0, 100)
y = list(i * i for i in x)
x_tensor = torch.FloatTensor(x).view(-1, 1)
y_tensor = torch.FloatTensor(y).view(-1, 1)

# 训练循环（5000次迭代）
for epoch in range(5000):
    pred = model(x_tensor)
    loss = criterion(pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# 测试
test_x = torch.FloatTensor(x).view(-1, 1)
test_y = torch.FloatTensor(y).view(-1, 1)
test_pred = model(test_x)
test_loss = criterion(test_pred, test_y)
print(f"Test Loss: {test_loss.item():.6f}")
