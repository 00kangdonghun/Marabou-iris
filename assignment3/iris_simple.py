import torch
import torch.nn as nn
import torch.onnx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터 로딩 및 전처리
iris = load_iris()
X = iris['data']
y = iris['target']
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 2. 간단한 MLP 모델 설계
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. 간단한 학습 (에폭 줄여 예시)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

# 4. ONNX로 변환
dummy_input = torch.randn(1, 4)
torch.onnx.export(model, dummy_input, "iris_simple.onnx", 
                  input_names=['input'], output_names=['output'])
