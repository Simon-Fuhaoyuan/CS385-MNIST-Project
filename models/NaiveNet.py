import torch
import torch.nn as nn


class NaiveNet(nn.Module):

    def __init__(self, features, opt, num_classes=10, init_weights=True):
        super(NaiveNet, self).__init__()
        self.opt = opt
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
        self.final_layer = None
        if self.opt.final_layer == 'softmax':
            self.final_layer = nn.Softmax(dim=1)
        else:
            self.final_layer = nn.Sigmoid()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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


def make_layers(cfg, opt, batch_norm=False):
    layers = []
    in_channels = opt.in_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = [32, 'M', 64, 'M']

def get_CNN(opt):
    model = NaiveNet(make_layers(cfg, opt, batch_norm=True), opt)

    return model