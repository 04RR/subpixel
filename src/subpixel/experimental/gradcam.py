from torchcam.methods import SmoothGradCAMpp
import cv2
import torch
from torchvision.transforms.functional import normalize
import numpy as np


def get_activationMap(model, image, device='cpu'):

    cam_extractor = SmoothGradCAMpp(model)

    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1).float()
        image = normalize(image / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    out = model(image.unsqueeze(0).to(device))

    return cam_extractor(out.squeeze(0).argmax().item(), out)


def get_bbox(activation_maps, image_shape= None, threshold=0.5):
    
    if image_shape:
        activation_maps = cv2.resize(activation_maps, image_shape)

    activation_map = activation_maps[0]
    activation_map = activation_map > threshold
    contours, _ = cv2.findContours(activation_map.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbox = cv2.boundingRect(contours[0])

    return bbox