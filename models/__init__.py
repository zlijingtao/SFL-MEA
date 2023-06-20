import sys
sys.path.append("../")
from models.resnet import ResNet18, ResNet34
from models.resnet_cifar import ResNet20, ResNet32
from models.mobilenetv2 import MobileNetV2
from models.vgg import vgg11, vgg13, vgg11_bn, vgg13_bn, vgg19_bn
import copy


def get_model(arch, cutting_layer, num_client, num_class, number_of_freeze_layer = 10):

    if arch == "resnet18":
        model = ResNet18(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "resnet20":
        model = ResNet20(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "resnet32":
        model = ResNet32(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "resnet34":
        model = ResNet34(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "vgg13":
        model = vgg13(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "vgg11":
        model = vgg11(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "vgg13_bn":
        model = vgg13_bn(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "vgg19_bn":
        model = vgg19_bn(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "vgg11_bn":
        model = vgg11_bn(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "mobilenetv2":
        model = MobileNetV2(cutting_layer, num_client=num_client, num_class=num_class)
    elif arch == "ViT":
        from transformers import ViTForImageClassification
        from models.vit_wrapper import vit_split_model_wrapper
        if num_class == 10:
            id2label = {0: 'airplane',
                        1: 'automobile',
                        2: 'bird',
                        3: 'cat',
                        4: 'deer',
                        5: 'dog',
                        6: 'frog',
                        7: 'horse',
                        8: 'ship',
                        9: 'truck'}
            label2id = {label:id for id,label in id2label.items()}
        else:
            raise("not yet support!")

        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                        id2label=id2label,
                                                        label2id=label2id)
        model = vit_split_model_wrapper(model, cutting_layer, num_client, number_of_freeze_layer)
    else:
        raise("No such architecture!")

    return model

