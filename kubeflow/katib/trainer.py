import torch
import torch.nn as nn
from torchvision import datasets, transforms
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    hyper_params_epochs: int = 5
    hyper_params_learning_rate: float = 0.01
    hyper_params_batch: int = 64


settings = Settings()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=settings.hyper_params_batch,
    shuffle=True,
)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=settings.hyper_params_learning_rate)


def train() -> float:
    model.train()
    total_loss, correct, total = 0, 0, 0
    for _ in range(settings.hyper_params_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Final Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss


if __name__ == "__main__":
    print("settings:", vars(settings))
    print()
    final_loss = train()

    # Log the final loss to a file
    with open("/output/metrics.log", "w") as f:
        f.write(f"loss={final_loss}\n")
