import torch
import torch.nn as nn
from torchvision import datasets, transforms
import mlflow.pytorch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    lr: float = 0.01
    nn_dim_input: int = 784
    nn_dim_hidden: int = 128
    nn_dim_output: int = 10
    batch_size: int = 64
    epochs: int = 5
    mlflow_tracking_uri: str = "http://0.0.0.0:5000"


settings = Settings()
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.pytorch.autolog()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(".", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=settings.batch_size,
    shuffle=True,
)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(settings.nn_dim_input, settings.nn_dim_hidden),
    nn.ReLU(),
    nn.Linear(settings.nn_dim_hidden, settings.nn_dim_output),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=settings.lr)


def train() -> None:
    model.train()
    for epoch in range(settings.epochs):
        total_loss, correct, total = 0, 0, 0
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

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total
        mlflow.log_metric("loss", avg_loss, step=epoch)
        mlflow.log_metric("accuracy", accuracy, step=epoch)


if __name__ == "__main__":
    with mlflow.start_run():
        mlflow.log_params(vars(settings))
        train()
