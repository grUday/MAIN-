import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the LSTM model
class LaneChangeModel(nn.Module):
    def __init__(self):
        super(LaneChangeModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

model = LaneChangeModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_samples = 600  # Ensuring equal 200 samples for each class

X_left = np.random.rand(200, 25, 4).astype(np.float32)
X_keep = np.random.rand(200, 25, 4).astype(np.float32)
X_right = np.random.rand(200, 25, 4).astype(np.float32)

X_train = np.vstack((X_left, X_keep, X_right))
y_train = np.array([0] * 200 + [1] * 200 + [2] * 200)  # Balanced dataset

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

torch.save(model.state_dict(), "lane_change_model.pth")
print("✅ Model Training Complete & Saved.")
