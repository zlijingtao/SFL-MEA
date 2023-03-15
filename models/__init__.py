import sys
sys.path.append("../")
from models.resnet import ResNet18, ResNet34
from models.resnet_cifar import ResNet20, ResNet32
from models.mobilenetv2 import MobileNetV2
from models.vgg import vgg11, vgg13, vgg11_bn, vgg13_bn

def get_model(arch, cutting_layer, logger, num_client, num_class):

    if arch == "resnet18":
        model = ResNet18(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "resnet20":
        model = ResNet20(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "resnet32":
        model = ResNet32(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "resnet34":
        model = ResNet34(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "vgg13":
        model = vgg13(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "vgg11":
        model = vgg11(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "vgg13_bn":
        model = vgg13_bn(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "vgg11_bn":
        model = vgg11_bn(cutting_layer, logger, num_client=num_client, num_class=num_class)
    elif arch == "mobilenetv2":
        model = MobileNetV2(cutting_layer, logger, num_client=num_client, num_class=num_class)
    else:
        raise("No such architecture!")

    return model