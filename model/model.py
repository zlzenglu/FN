
from .ResNet_Zoo import ResNet, BasicBlock, ResNet101, ResNet152,ResNet50,ResNet50mini
from .models import LeNet5,VGG11


def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def resnet18gray(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, num_channels=1)

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def resnet50(num_classes=10):
    return ResNet50(num_classes=num_classes)

def resnet50mini(num_classes=10):
    return ResNet50mini(num_classes=num_classes)

def resnet101(num_classes=10):
    return ResNet101(num_classes=num_classes)

def resnet152(num_classes=10):
    return ResNet152(num_classes=num_classes)

def lenet5(num_classes=10):
    return LeNet5(num_classes=num_classes)

def vgg11(input_channels=3,num_classes=10):
    return VGG11(input_channels=input_channels,num_classes=num_classes)

def vgg16gray(input_channels=1,num_classes=10):
    return VGG11(input_channels=input_channels,num_classes=num_classes)
