import torch
import torch.nn as nn


class LinearNet(nn.Module):

    def __init__(self, opt, num_classes=10, init_weights=True):
        super(LinearNet, self).__init__()
        self.opt = opt
        self.classifier = nn.Linear(self.opt.in_channel * 28 * 28, num_classes)
        self.final_layer = None
        if self.opt.final_layer == 'softmax':
            self.final_layer = nn.Softmax(dim=1)
        else:
            self.final_layer = nn.Sigmoid()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.opt.loss != 'crossentropy':
            x = self.final_layer(x)
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


def get_CNN(opt):
    model = LinearNet(opt)

    return model