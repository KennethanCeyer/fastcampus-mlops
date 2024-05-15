import torch
import torchvision.models as models

def model_fn(model_dir):
    model = models.resnet18()
    model.load_state_dict(torch.load(model_dir + "/model.pth"))
    model.eval()
    return model

