import torch
from torchvision.models import resnet50

def get_pretrained_resnet50(device):
    model = resnet50(pretrained=True)
    model.eval()
    model.to(device)
    return model

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
