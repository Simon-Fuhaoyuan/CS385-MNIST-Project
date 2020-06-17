import torch
import torch.nn as nn


class SmallNet(nn.Module):

    def __init__(self, opt, num_classes=10, init_weights=True):
        super(SmallNet, self).__init__()
        self.opt = opt
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(opt.in_channel, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)
        self.final_layer = None
        if self.opt.final_layer == 'softmax':
            self.final_layer = nn.Softmax(dim=1)
        else:
            self.final_layer = nn.Sigmoid()
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        feature0 = torch.flatten(x, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        feature1 = torch.flatten(x, 1)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        feature2 = torch.flatten(x, 1)

        fmaps = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature3 = torch.flatten(x, 1)

        x = self.classifier(x)

        if self.opt.loss != 'crossentropy':
            x = self.final_layer(x)
        return x, [feature0, feature1, feature2, feature3]

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
    model = SmallNet(opt)

    return model