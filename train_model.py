import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load MNIST
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data', train=False, download=True,
        transform=transforms.ToTensor()), batch_size=1000)

# Define your model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100

    print(f"Epoch {epoch + 1:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), 'model.pth')
print("\n Model training complete and saved to 'model.pth'")
