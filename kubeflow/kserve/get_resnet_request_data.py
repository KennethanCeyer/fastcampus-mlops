from torchvision import transforms
from torchvision import datasets
import json


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

cifar10 = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
image, label = cifar10[0]
image = image.numpy()

input_data = {"instances": image.tolist()}

with open("input.json", "w") as f:
    json.dump(input_data, f)
