import math
from collections import OrderedDict

import json
import torch
import torch.nn.functional as F
from torch import nn

import torch.utils.checkpoint as cp
from torch.optim import SGD, Adam
from torch.utils import model_zoo
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import models
from torchvision.models import resnet18

from dataset import cifar100


class LeNet(nn.Module):

    def __init__(self, channel=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten())
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]

    def forward(self, x, start_layer_idx=0):
        '''
        One forward pass through the network.

        Args:
            x: input
        '''
        if start_layer_idx < 0:
            start_layer_idx = len(self.layers) + start_layer_idx

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


class Net(nn.Module):
    def __init__(self, channel=1, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Flatten())
        self.fc1 = nn.Sequential(nn.Linear(64 * 7 * 7, 128), nn.ReLU())
        # self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            start_layer_idx = len(self.layers) + start_layer_idx

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetV2(nn.Module):
    def __init__(self, block, num_blocks, channel, num_classes=10):
        super(ResNetV2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)
        self.layers = [nn.Sequential(self.conv1, self.bn1, nn.ReLU()), self.layer1, self.layer2, self.layer3,
                       nn.Sequential(self.avgpool, nn.Flatten()), self.linear]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            start_layer_idx = len(self.layers) + start_layer_idx

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


def ResNet20(channel=3, num_classes=10):
    return ResNetV2(ResidualBlock, [3, 3, 3], channel=3, num_classes=num_classes)


def ResNet32(channel=3, num_classes=10):
    # https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    return ResNetV2(ResidualBlock, [5, 5, 5], channel=3, num_classes=num_classes)


def ResNet56(channel=3, num_classes=10):
    # https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    return ResNetV2(ResidualBlock, [9, 9, 9], channel=3, num_classes=num_classes)


import torch.nn as nn


class MobileNetV2CifarBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileNetV2CifarBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, channel=3, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block_layer = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(4, 1)
        self.layers = [nn.Sequential(self.conv1, self.bn1, self.relu), self.block_layer,
                       nn.Sequential(self.conv2, self.bn2, self.relu),
                       nn.Sequential(self.avgpool, nn.Flatten()), self.linear]

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(MobileNetV2CifarBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0:
            start_layer_idx = len(self.layers) + start_layer_idx

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Norm2d(planes, norm='bn'):
    if norm == 'gn':
        return nn.GroupNorm(planes // 16, planes)
    elif norm == 'ln':
        return nn.GroupNorm(1, planes)
    elif norm == 'bn':
        return nn.BatchNorm2d(planes)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample=None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation=None,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.layers = [nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool), self.layer1, self.layer2,
                       self.layer3, self.layer4, nn.Sequential(self.avgpool, nn.Flatten()), self.fc]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx=0):
        start_layer_idx = (len(self.layers) + start_layer_idx) % len(self.layers)

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


def ResNet18(channel, num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm_layer=Norm2d)


class CIFAR100Model(nn.Module):
    def __init__(self, channel=3, num_classes=100):
        super().__init__()

        self.conv1 = nn.Conv2d(channel, 128, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, 128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, 256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)  # 384
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.layers = [nn.Sequential(self.conv1, self.relu, self.gn1), nn.Sequential(self.conv2, self.relu, self.gn2),
                       nn.Sequential(self.conv3, self.relu, self.pool, self.dropout),
                       nn.Sequential(nn.Flatten(), self.fc1, self.relu, self.dropout),
                       self.fc2]

    def forward(self, x, start_layer_idx=0):
        start_layer_idx = (len(self.layers) + start_layer_idx) % len(self.layers)

        feature = None
        for i in range(start_layer_idx, len(self.layers)):
            x = self.layers[i](x)
            if i == len(self.layers) - 2:
                feature = x

        return feature, x


def test_model():
    def test_acc(model, test_loader, criterion):
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                feature, outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            print(
                f'Accuracy of the model on the test images: {100 * correct / total} %, loss: {criterion(outputs, labels)}')
        return correct / total

    X, y, test_loader = cifar100()
    num_epochs = 60
    learning_rate = 0.05
    batch_size = 64
    device = "cuda"
    model = CIFAR100Model()
    model = model.to("cuda")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_loader = DataLoader(dataset=list(zip(X, y)), batch_size=batch_size,
                              sampler=SubsetRandomSampler(list(range(len(X)))))

    accs = []
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            feature, outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        acc = test_acc(model, test_loader, criterion)
        accs.append(acc)

    with open("singe_test.json", "w") as f:
        json.dump({"accs": accs}, f)


def count_trainable_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Parameters: {param.numel()}")
            total_params += param.numel()
    print(f"Total trainable parameters: {total_params}")


if __name__ == '__main__':
    # model = ResNet20(3, 10)
    # model = Net()
    model = CIFAR100Model()
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    print_model_parameters(model)
