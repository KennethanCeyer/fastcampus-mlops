import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
torch.save(model.state_dict(), "model.pth")

