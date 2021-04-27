import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from numpy import random
from utils.util import save_feature




__all__ = ['AlexNet', 'AlexNet_pretrain']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet(nn.Module):

    def __init__(self, saveFeature = False, cfg = None):
        super(AlexNet, self).__init__()
        self.saveFeature = saveFeature
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, cfg["num_class"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.saveFeature:
            save_feature(x, "alexnet", "conv1")
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def AlexNet_pretrain(saveFeature = False,cfg = None):
    model = AlexNet(saveFeature = saveFeature, cfg = cfg)
    num_class = cfg["num_class"]
    state_dict = load_state_dict_from_url(model_urls['alexnet'])
    if num_class != 1000:
        weight = torch.Tensor(random.rand(num_class, 4096))
        bias = torch.Tensor(random.rand(num_class))
        state_dict["classifier.6.weight"] = weight
        state_dict["classifier.6.bias"] = bias
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    cfg = dict(
        AlexNet = {
            "num_class":10,
        }
    )
    model = AlexNet_pretrain(cfg = cfg)