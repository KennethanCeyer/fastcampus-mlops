import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform: transforms.Compose = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)

trainset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader: DataLoader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader: DataLoader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

class CNNNetwork(nn.Module):
    def __init__(self) -> None:
        super(CNNNetwork, self).__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3: nn.Conv2d = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        self.fc1: nn.Linear = nn.Linear(256 * 4 * 4, 1024)
        self.fc2: nn.Linear = nn.Linear(1024, 10)
        self.dropout: nn.Dropout = nn.Dropout(0.5)
        self.relu: nn.ReLU = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model: CNNNetwork = CNNNetwork().to(device)
criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss: float = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

correct: int = 0
total: int = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

