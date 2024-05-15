from ts.torch_handler.base_handler import BaseHandler
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

class MyHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, data):
        image = data[0].get("data") or data[0].get("body")
        image = Image.open(io.BytesIO(image)).convert('RGB')
        transformation = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transformation(image).unsqueeze(0).to(self.device)

    def inference(self, data, *args, **kwargs):
        with torch.no_grad():
            outputs = self.model(data)
        return outputs.argmax(1).tolist()

    def postprocess(self, data):
        return data

