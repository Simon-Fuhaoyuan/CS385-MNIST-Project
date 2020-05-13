import torch
import torch.nn as nn


class LinearNet(nn.Module):

    def __init__(self, num_classes=10, init_weights=True):
        super(LinearNet, self).__init__()
        self.classifier = nn.Linear(1 * 28 * 28, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_CNN(in_channels=1):
    model = LinearNet()

    return model