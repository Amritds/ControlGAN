import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from miscc.config import cfg


class resnet_encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(resnet_encoder, self).__init__()
      
        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        modules = list(resnet.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.resnet = nn.Sequential(*modules)
        
        for p in self.resnet.parameters():
            p.requires_grad = False
        

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        
        return torch.flatten(self.resnet(images), 1)