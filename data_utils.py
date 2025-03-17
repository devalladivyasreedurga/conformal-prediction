import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

def get_imagenet_val_loader(root, batch_size=32, num_workers=4):
    """
    Loads the ImageNet validation set from 'root',
    which should contain subfolders named by WordNet IDs.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=root, transform=transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return loader

import numpy as np
from tqdm import tqdm

def compute_logits_and_labels(model, loader, device):
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)

def split_calib_test(logits, labels):
    n = len(labels)
    n_calib = n // 2
    calib_logits = logits[:n_calib]
    calib_labels = labels[:n_calib]
    test_logits = logits[n_calib:]
    test_labels = labels[n_calib:]
    return calib_logits, calib_labels, test_logits, test_labels
