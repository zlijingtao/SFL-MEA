# import tensorflow as tf
# import numpy as np
import functools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.nn.modules.linear import Linear





def get_decoder(gan_AE_type, input_nc, output_nc, input_dim, output_dim, gan_AE_activation):
    if gan_AE_type == "custom":
        decoder = custom_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                            activation=gan_AE_activation).cuda()
    elif gan_AE_type == "custom_bn":
        decoder = custom_AE_bn(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                                activation=gan_AE_activation).cuda()
    elif gan_AE_type == "complex":
        decoder = complex_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                            activation=gan_AE_activation).cuda()
    elif gan_AE_type == "complex_plus":
        decoder = complex_plus_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim,
                                                output_dim=output_dim, activation=gan_AE_activation).cuda()
    elif gan_AE_type == "complex_res":
        decoder = complex_res_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim,
                                                output_dim=output_dim, activation=gan_AE_activation).cuda()
    elif gan_AE_type == "complex_resplus":
        decoder = complex_resplus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                    output_dim=output_dim, activation=gan_AE_activation).cuda()
    elif "complex_resplusN" in gan_AE_type:
        try:
            N = int(gan_AE_type.split("complex_resplusN")[1])
        except:
            print("auto extract N from complex_resplusN failed, set N to default 2")
            N = 2
        decoder = complex_resplusN_AE(N = N, input_nc=input_nc, output_nc=output_nc, input_dim=input_dim,
                                                        output_dim=output_dim, activation=gan_AE_activation).cuda()
    elif "complex_normplusN" in gan_AE_type:
        try:
            afterfix = gan_AE_type.split("complex_normplusN")[1]
            N = int(afterfix.split("C")[0])
            internal_C = int(afterfix.split("C")[1])
        except:
            print("auto extract N from complex_normplusN failed, set N to default 2")
            N = 0
            internal_C = 64
        decoder = complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=output_nc,
                                                    input_dim=input_dim, output_dim=output_dim,
                                                    activation=gan_AE_activation).cuda()
    
    elif "conv_normN" in gan_AE_type:
        try:
            afterfix = gan_AE_type.split("conv_normN")[1]
            N = int(afterfix.split("C")[0])
            internal_C = int(afterfix.split("C")[1])
        except:
            print("auto extract N from conv_normN failed, set N to default 2")
            N = 0
            internal_C = 64
        decoder = conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=output_nc,
                                                    input_dim=input_dim, output_dim=output_dim,
                                                    activation=gan_AE_activation).cuda()

    elif "res_normN" in gan_AE_type:
        try:
            afterfix = gan_AE_type.split("res_normN")[1]
            N = int(afterfix.split("C")[0])
            internal_C = int(afterfix.split("C")[1])
        except:
            print("auto extract N from res_normN failed, set N to default 2")
            N = 0
            internal_C = 64
        decoder = res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=output_nc,
                                                    input_dim=input_dim, output_dim=output_dim,
                                                    activation=gan_AE_activation).cuda()
    
    elif "TB_normplusN" in gan_AE_type:
        try:
            afterfix = gan_AE_type.split("TB_normplusN")[1]
            N = int(afterfix.split("C")[0])
            internal_C = int(afterfix.split("C")[1])
        except:
            print("auto extract N from TB_normplusN failed, set N to default 0")
            N = 0
            internal_C = 64
        decoder = TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=output_nc,
                                                    input_dim=input_dim, output_dim=output_dim,
                                                    activation=gan_AE_activation).cuda()
    elif gan_AE_type == "simple":
        decoder = simple_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                            activation=gan_AE_activation).cuda()
    elif gan_AE_type == "simple_bn":
        decoder = simple_AE_bn(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                                activation=gan_AE_activation).cuda()
    elif gan_AE_type == "simplest":
        decoder = simplest_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=output_dim,
                                            activation=gan_AE_activation).cuda()
    else:
        raise ("No such GAN AE type.")
    return decoder


def create_surrogate_model(arch, cutting_layer, num_class, train_clas_layer = 0, surrogate_arch = "same"):
    train_clas_layer = int(train_clas_layer)

    key_dict = { "resnet18": "A", "resnet34": "B", "resnet20": "C", "resnet32": "D", 
    "vgg11_bn": "A", "vgg11": "A", "vgg13_bn": "B", "vgg13": "B", "vgg16_bn": "D", "vgg16": "D", "vgg19_bn": "E", "vgg19": "E", 

    }

    if "vgg" in arch:
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                512, 512, 512, 512, 'M'],
            'fc': [512, 512, 512],
        }

        if surrogate_arch == "shorter":
            cfg['fc'] = cfg['fc'][1:]
            print("shorter. CFG is {}".format(str(cfg)))
        elif surrogate_arch == "longer":
            cfg['fc'].append(512)
            print("longer. CFG is {}".format(str(cfg)))
        
        if train_clas_layer <= len(cfg['fc']):
            if surrogate_arch == "wider":
                for i in range(train_clas_layer - 1):
                    cfg['fc'][len(cfg['fc']) - 1 - i] = 2 * cfg['fc'][len(cfg['fc']) - 1 - i]
                print("wider. CFG is {}".format(str(cfg)))
            elif surrogate_arch == "thinner":
                for i in range(train_clas_layer - 1):
                    cfg['fc'][len(cfg['fc']) - 1 - i] = int(1/2 * cfg['fc'][len(cfg['fc']) - 1 - i])
                print("tinner. CFG is {}".format(str(cfg)))
        else:
            if surrogate_arch == "wider":
                mul_factor = 2
            elif surrogate_arch == "thinner":
                mul_factor = 0.5
            else:
                mul_factor = 1
            for i in range(len(cfg['fc'])):
                cfg['fc'][len(cfg['fc']) - 1 - i] = int(mul_factor * cfg['fc'][len(cfg['fc']) - 1 - i])
            # need to separate "M" out:
            # cfga_range = cfg['A'][-(train_clas_layer - len(cfg['fc'])):].count("M")
            j = 1
            count = 0
            while(count < train_clas_layer - len(cfg['fc'])):
                if cfg[key_dict[arch]][-j] != "M":
                    cfg[key_dict[arch]][-j] = int(mul_factor * cfg[key_dict[arch]][-j])
                    count += 1
                j += 1


            # for j in range(cfga_range):
            #     if cfg['A'][len(cfg['A']) - 1 - j] != "M":
            #         cfg['A'][len(cfg['A']) - 1 - j] = int(mul_factor * cfg['A'][len(cfg['A']) - 1 - j]) 
            print("{}. CFG is {}".format(surrogate_arch, str(cfg)))
            
        if arch == "vgg11_bn":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['A'], batch_norm=True), cfg['fc'], num_class = num_class)
        elif arch == "vgg11":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['A'], batch_norm=False), cfg['fc'], num_class = num_class)
        elif arch == "vgg13_bn":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['B'], batch_norm=True), cfg['fc'], num_class = num_class)
        elif arch == "vgg13":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['B'], batch_norm=False), cfg['fc'], num_class = num_class)
        elif arch == "vgg16_bn":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['D'], batch_norm=True), cfg['fc'], num_class = num_class)
        elif arch == "vgg16":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['D'], batch_norm=False), cfg['fc'], num_class = num_class)
        elif arch == "vgg19_bn":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['E'], batch_norm=True), cfg['fc'], num_class = num_class)
        elif arch == "vgg19":
            return VGG_surrogate(make_vgg_layers(cutting_layer,cfg['E'], batch_norm=False), cfg['fc'], num_class = num_class)
    elif "resnet" in arch:
        cfg = {
            'A': [64, 64, 64, 128, 128, 256, 256, 512, 512],
            'B': [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512],
            'C': [16, 16, 16, 16, 32, 32, 32, 64, 64, 64],
            'D': [16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64],
        }
        if surrogate_arch == "longer":
            print("Longer Archtecture in Resnet is not supported yet, change to default same architecture")
            surrogate_arch = "same"
        elif surrogate_arch == "shorter":
            print("Shorter Archtecture in Resnet is not supported yet, change to default same architecture")
            surrogate_arch = "same"
        
        if train_clas_layer > 1:
            if surrogate_arch == "wider":
                mul_factor = 2
            elif surrogate_arch == "thinner":
                mul_factor = 0.5
            else:
                mul_factor = 1
            j = 1
            count = 0
            while(count < train_clas_layer - 1):
                cfg[key_dict[arch]][-j] = int(mul_factor * cfg[key_dict[arch]][-j])
                count += 1
                j += 1
        if arch == "resnet18":
            return ResNet_surrogate(make_ResNet_layers(BasicBlock, [2, 2, 2, 2], cfg['A'], cutting_layer), num_class = num_class, fc_size = cfg['A'][-1])
        elif arch == "resnet34":
            return ResNet_surrogate(make_ResNet_layers(BasicBlock, [3, 4, 6, 3], cfg['B'], cutting_layer), num_class = num_class, fc_size = cfg['B'][-1])
        elif arch == "resnet20":
            return cifarResNet_surrogate(make_cifarResNet_layers(cifarResNet_BasicBlock, [3, 3, 3], cfg['C'], cutting_layer), num_class = num_class, fc_size = cfg['C'][-1])
        elif arch == "resnet32":
            return cifarResNet_surrogate(make_cifarResNet_layers(cifarResNet_BasicBlock, [5, 5, 5], cfg['D'], cutting_layer), num_class = num_class, fc_size = cfg['D'][-1])
    elif "mobilenetv2" in arch:
        cfg = {"A": [32, [1,  16, 1, 1],
           [6,  24, 2, 1],  # NOTE: change stride 2 -> 1 for CIFAR10
           [6,  32, 3, 2],
           [6,  64, 4, 2],
           [6,  96, 3, 1],
           [6, 160, 3, 2],
           [6, 320, 1, 1], 1280]}
        if surrogate_arch == "longer":
            print("Longer Archtecture in mobilenetv2 is not supported yet, change to default same architecture")
            surrogate_arch = "same"
        elif surrogate_arch == "shorter":
            print("Shorter Archtecture in mobilenetv2 is not supported yet, change to default same architecture")
            surrogate_arch = "same"
        if train_clas_layer > 1:
            if surrogate_arch == "wider":
                mul_factor = 2
            elif surrogate_arch == "thinner":
                mul_factor = 0.5
            else:
                mul_factor = 1
            j = 1
            count = 0
            while(count < train_clas_layer - 1):
                if isinstance(cfg["A"][-j], list):
                    cfg["A"][-j][1] = int(mul_factor * cfg["A"][-j][1])
                else:
                    cfg["A"][-j] = int(mul_factor * cfg["A"][-j])
                count += 1
                j += 1


        return MobileNet_surrogate(make_mobilenet_layers(cutting_layer,cfg["A"], in_planes=32), num_class = num_class)
    elif "ViT" in arch:
        if surrogate_arch != "same":
            raise("other surrogate arc options are not supported")
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
        model = vit_split_model_wrapper(model, 12 - train_clas_layer, 1, 12 - train_clas_layer)
        return model


class MobView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = F.avg_pool2d(input, 4)
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = out.view(shape)
        return out



class MobileNet_surrogate(nn.Module):
    

    def __init__(self, feature, num_class = 10):
        super(MobileNet_surrogate, self).__init__()
        self.local = feature[0]
        self.cloud = feature[1]
        self.length_tail = feature[2]
        self.classifier = nn.Linear(1280, num_class)
        self.length_clas = 1
        self.cloud_classifier_merge = False
        self.original_num_cloud = self.get_num_of_cloud_layer()

    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(MobView())
        cloud_list.append(self.classifier)
        self.cloud = nn.Sequential(*cloud_list)

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "MobView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 1
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer

    def recover(self):
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()
            

    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
            
        list_of_layers = list(self.local.children())
        list_of_layers.extend(list(self.cloud.children()))
        total_layer = 0
        for _, module in enumerate(list_of_layers):
            if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                total_layer += 1
        num_of_local_layer = (total_layer - num_of_cloud_layer)
        local_list = []
        local_count = 0
        cloud_list = []
        for _, module in enumerate(list_of_layers):
            if ("Conv2d" in str(module) and "Block" not in str(module)) or "Linear" in str(module) or "Block" in str(module):
                local_count += 1
            if local_count <= num_of_local_layer:
                local_list.append(module)
            else:
                cloud_list.append(module)
        
        self.cloud = nn.Sequential(*cloud_list)
        self.local = nn.Sequential(*local_list)

    
    
    
    
    def forward(self, x):
        if self.cloud_classifier_merge:
            x = self.local(x)
            x = self.cloud(x)
        else:
            local_output = self.local(x)
            x = self.cloud(local_output)
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x


class mobilenet_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(mobilenet_Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

def make_mobilenet_layers(cutting_layer, cfg, in_planes):
        local_layer_list = []
        cloud_layer_list = []
        current_layer = 0
        length_tail = 0
        if cutting_layer > current_layer:
            local_layer_list.append(nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False))
            local_layer_list.append(nn.BatchNorm2d(32))
        else:
            cloud_layer_list.append(nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False))
            cloud_layer_list.append(nn.BatchNorm2d(32))
            length_tail += 1
        
        for expansion, out_planes, num_blocks, stride in cfg[1:-1]:
            current_layer += 1
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                if cutting_layer > current_layer:
                    local_layer_list.append(mobilenet_Block(in_planes, out_planes, expansion, stride))
                else:
                    cloud_layer_list.append(mobilenet_Block(in_planes, out_planes, expansion, stride))
                    length_tail += 1
                in_planes = out_planes
        current_layer += 1
        if cutting_layer > current_layer:
            local_layer_list.append(nn.Conv2d(cfg[-2][1], cfg[-1], kernel_size=1, stride=1, padding=0, bias=False))
            local_layer_list.append(nn.BatchNorm2d(1280))
        else:
            cloud_layer_list.append(nn.Conv2d(cfg[-2][1], cfg[-1], kernel_size=1, stride=1, padding=0, bias=False))
            cloud_layer_list.append(nn.BatchNorm2d(1280))
            length_tail += 1

        return nn.Sequential(*local_layer_list), nn.Sequential(*cloud_layer_list), length_tail





class CifarResView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = F.avg_pool2d(input, 8)
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = out.view(shape)
        return out

class VGGView(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, -1)
        out = input.view(shape)
        return out

class VGG_surrogate(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, feature, fc_cfg, num_class = 10, fc_size = 512):
        super(VGG_surrogate, self).__init__()
        self.local = feature[0]
        self.cloud = feature[1]
        self.length_tail = feature[2]
        self.cloud_classifier_merge = False
        classifier_list = []

        for i in range(len(fc_cfg) - 1):
            classifier_list += [nn.Dropout(), nn.Linear(fc_cfg[i], fc_cfg[i+1]), nn.ReLU(True)]
        
        classifier_list += [nn.Linear(fc_cfg[-1], num_class)]
        self.classifier = nn.Sequential(*classifier_list)
        self.length_clas = len(fc_cfg)
        self.fc_cfg = fc_cfg
        self.original_num_cloud = self.get_num_of_cloud_layer()

    def forward(self, x):
        if self.cloud_classifier_merge:
            x = self.local(x)
            x = self.cloud(x)
        else:
            self.local_output = self.local(x)
            x = self.cloud(self.local_output)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(VGGView())
        classifier_list = list(self.classifier.children())
        cloud_list.extend(classifier_list)
        self.cloud = nn.Sequential(*cloud_list)
        self.length_clas = 0

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "VGGView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)
        self.length_clas = len(self.fc_cfg)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "Conv2d" in str(module) or "Linear" in str(module) or "MaxPool2d" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += self.length_clas
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "Conv2d" in str(module) or "Linear" in str(module) or "MaxPool2d" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer
    
    def recover(self):
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()

    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
        list_of_layers = list(self.local.children())
        list_of_layers.extend(list(self.cloud.children()))
        total_layer = 0
        for i, module in enumerate(list_of_layers):
            if "Conv2d" in str(module) or "Linear" in str(module) or "MaxPool2d" in str(module):
                total_layer += 1
        
        num_of_local_layer = (total_layer - num_of_cloud_layer)
        local_list = []
        local_count = 0
        cloud_list = []
        for i, module in enumerate(list_of_layers):
            if "Conv2d" in str(module) or "Linear" in str(module) or "MaxPool2d" in str(module):
                local_count += 1
            if local_count <= num_of_local_layer:
                local_list.append(module)
            else:
                cloud_list.append(module)
        
        self.cloud = nn.Sequential(*cloud_list)
        self.local = nn.Sequential(*local_list)

        self.length_tail = num_of_cloud_layer



def make_vgg_layers(cutting_layer,cfg, batch_norm=False):
    local = []
    cloud = []
    in_channels = 3
    length_tail = 0
    #Modified Local part - Experimental feature
    channel_mul = 1
    for v_idx,v in enumerate(cfg):
        if v_idx < cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, int(v * channel_mul), kernel_size=3, padding=1)
                if batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = int(v * channel_mul)
        elif v_idx == cutting_layer - 1:
            if v == 'M':
                local += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    local += [conv2d, nn.BatchNorm2d(int(v * channel_mul)), nn.ReLU(inplace=True)]
                else:
                    local += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        else:
            if v == 'M':
                cloud += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    cloud += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    cloud += [conv2d, nn.ReLU(inplace=True)]
                length_tail += 1
                in_channels = v

    return nn.Sequential(*local), nn.Sequential(*cloud), length_tail



class ResNet_surrogate(nn.Module):
    '''
    ResNet model 
    '''
    def __init__(self, feature, expansion = 1, num_class = 10, fc_size = 512):
        super(ResNet_surrogate, self).__init__()
        self.local = feature[0]
        self.cloud = feature[1]
        self.length_tail = feature[2]
        self.classifier = nn.Linear(fc_size*expansion, num_class)
        self.length_clas = 1
        self.cloud_classifier_merge = False
        self.original_num_cloud = self.get_num_of_cloud_layer()

    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(MobView())
        cloud_list.append(self.classifier)
        self.cloud = nn.Sequential(*cloud_list)

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "MobView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += 1
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer

    def recover(self):
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()
            

    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
        
        list_of_layers = list(self.local.children())
        list_of_layers.extend(list(self.cloud.children()))
        total_layer = 0
        for _, module in enumerate(list_of_layers):
            if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module):
                total_layer += 1
        num_of_local_layer = (total_layer - num_of_cloud_layer)
        local_list = []
        local_count = 0
        cloud_list = []
        for _, module in enumerate(list_of_layers):
            if "conv3x3" in str(module) or "Linear" in str(module) or "BasicBlock" in str(module):
                local_count += 1
            if local_count <= num_of_local_layer:
                local_list.append(module)
            else:
                cloud_list.append(module)
        
        self.cloud = nn.Sequential(*cloud_list)
        self.local = nn.Sequential(*local_list)
    def forward(self, x):
        if self.cloud_classifier_merge:
            x = self.local(x)
            x = self.cloud(x)
        else:
            self.local_output = self.local(x)
            x = self.cloud(self.local_output)
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

class cifarResNet_surrogate(nn.Module):
    '''
    cifarResNet model 
    '''
    def __init__(self, feature, expansion = 1, num_class = 10, fc_size = 512):
        super(cifarResNet_surrogate, self).__init__()
        self.local = feature[0]
        self.cloud = feature[1]
        self.length_tail = feature[2]
        self.classifier = nn.Linear(fc_size*expansion, num_class)
        self.length_clas = 1
        self.cloud_classifier_merge = False
        self.original_num_cloud = self.get_num_of_cloud_layer()
    def merge_classifier_cloud(self):
        self.cloud_classifier_merge = True
        cloud_list = list(self.cloud.children())
        cloud_list.append(CifarResView())
        # classifier_list = list(self.classifier.children())
        cloud_list.append(self.classifier)
        self.cloud = nn.Sequential(*cloud_list)

    def unmerge_classifier_cloud(self):
        self.cloud_classifier_merge = False
        cloud_list = list(self.cloud.children())
        orig_cloud_list = []
        for i, module in enumerate(cloud_list):
            if "CifarResView" in str(module):
                break
            else:
                orig_cloud_list.append(module)
        self.cloud = nn.Sequential(*orig_cloud_list)

    def get_num_of_cloud_layer(self):
        num_of_cloud_layer = 0
        if not self.cloud_classifier_merge:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "ResNet_BasicBlock" in str(module):
                    num_of_cloud_layer += 1
            num_of_cloud_layer += self.length_clas
        else:
            list_of_layers = list(self.cloud.children())
            for i, module in enumerate(list_of_layers):
                if "conv3x3" in str(module) or "Linear" in str(module) or "ResNet_BasicBlock" in str(module):
                    num_of_cloud_layer += 1
        return num_of_cloud_layer
    
    def recover(self):
        if self.cloud_classifier_merge:
            self.resplit(self.original_num_cloud)
            self.unmerge_classifier_cloud()
                

    def resplit(self, num_of_cloud_layer):
        if not self.cloud_classifier_merge:
            self.merge_classifier_cloud()
            
        list_of_layers = list(self.local.children())
        list_of_layers.extend(list(self.cloud.children()))
        total_layer = 0
        for _, module in enumerate(list_of_layers):
            if "conv3x3" in str(module) or "Linear" in str(module) or "ResNet_BasicBlock" in str(module):
                total_layer += 1
        
        num_of_local_layer = (total_layer - num_of_cloud_layer)
        local_list = []
        local_count = 0
        cloud_list = []
        for _, module in enumerate(list_of_layers):
            if "conv3x3" in str(module) or "Linear" in str(module) or "ResNet_BasicBlock" in str(module):
                local_count += 1
            if local_count <= num_of_local_layer:
                local_list.append(module)
            else:
                cloud_list.append(module)
        
        self.cloud = nn.Sequential(*cloud_list)
        self.local = nn.Sequential(*local_list)
    def forward(self, x):
        if self.cloud_classifier_merge:
            x = self.local(x)
            x = self.cloud(x)
        else:
            self.local_output = self.local(x)
            x = self.cloud(self.local_output)
            x = F.avg_pool2d(x, 8)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x

class conv3x3(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out

def make_ResNet_layers(block, layer_list, cfgA, cutting_layer):

    layers = []
    current_image_dim = 32
    count = 0
    layers.append(conv3x3(3, cfgA[count]))
    in_planes = cfgA[count]
    count += 1

    strides = [1] + [1]*(layer_list[0]-1)
    for stride in strides:
        layers.append(block(in_planes, cfgA[count], stride))
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1

    strides = [2] + [1]*(layer_list[1]-1)
    for stride in strides:
        layers.append(block(in_planes, cfgA[count], stride))
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1

    strides = [2] + [1]*(layer_list[2]-1)
    for stride in strides:
        layers.append(block(in_planes, cfgA[count], stride))
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1

    strides = [2] + [1]*(layer_list[3]-1)
    for stride in strides:
        layers.append(block(in_planes, cfgA[count], stride))
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1
    try:
        local_layer_list = layers[:cutting_layer]
        cloud_layer_list = layers[cutting_layer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []
    length_tail = 2 * len(cloud_layer_list)
    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)

    return local, cloud, length_tail


class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)

def make_cifarResNet_layers(block, layer_list, cfgA, cutting_layer):

    layers = []
    current_image_dim = 32
    count = 0
    layers.append(conv3x3(3, cfgA[count]))
    in_planes = cfgA[count]
    count += 1
    downsample = None

    strides = [1] + [1]*(layer_list[0]-1)
    for i, stride in enumerate(strides):
        if stride != 1 or in_planes != cfgA[count] * block.expansion:
            downsample = DownsampleA(in_planes, cfgA[count] * block.expansion, stride)
        if i == 0:
            layers.append(block(in_planes, cfgA[count], stride, downsample))
        else:
            layers.append(block(in_planes, cfgA[count], stride))
        current_image_dim = current_image_dim // stride

        in_planes = cfgA[count] * block.expansion
        count += 1

    strides = [2] + [1]*(layer_list[1]-1)
    for i, stride in enumerate(strides):
        if stride != 1 or in_planes != cfgA[count] * block.expansion:
            downsample = DownsampleA(in_planes, cfgA[count] * block.expansion, stride)
        if i == 0:
            layers.append(block(in_planes, cfgA[count], stride, downsample))
        else:
            layers.append(block(in_planes, cfgA[count], stride))
        
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1

    strides = [2] + [1]*(layer_list[2]-1)
    for i, stride in enumerate(strides):
        if stride != 1 or in_planes != cfgA[count] * block.expansion:
            downsample = DownsampleA(in_planes, cfgA[count] * block.expansion, stride)
        if i == 0:
            layers.append(block(in_planes, cfgA[count], stride, downsample))
        else:
            layers.append(block(in_planes, cfgA[count], stride))
        
        current_image_dim = current_image_dim // stride
        in_planes = cfgA[count] * block.expansion
        count += 1
    try:
        local_layer_list = layers[:cutting_layer]
        cloud_layer_list = layers[cutting_layer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []
    length_tail = len(cloud_layer_list)
    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)

    return local, cloud, length_tail

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out




class cifarResNet_BasicBlock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(cifarResNet_BasicBlock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        return F.relu(residual + basicblock, inplace=True)

class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, final_bn=True):
        super(GeneratorA, self).__init__() 

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False) 
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                nn.Tanh()
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
        # if pre_x :
        #     return img
        # else:
        #     # img = nn.functional.interpolate(img, scale_factor=2)
        #     return self.activation(img)



class GeneratorC(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(GeneratorC, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

# class GeneratorC_single(nn.Module):
#     '''
#     Conditional Generator
#     '''
#     def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
#         super(GeneratorC_single, self).__init__()
        
#         self.init_size = img_size//4
#         self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

#         self.conv_blocks0 = nn.Sequential(
#             nn.BatchNorm2d(ngf*2),
#         )
#         self.conv_blocks1 = nn.Sequential(
#             nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.conv_blocks2 = nn.Sequential(
#             nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
#             nn.BatchNorm2d(ngf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
#             nn.Tanh(),
#             nn.BatchNorm2d(nc, affine=False) 
#         )

#     def forward(self, z):
#         # Concatenate label embedding and image to produce input
#         out = self.l1(z.view(z.shape[0],-1))
#         out = out.view(out.shape[0], -1, self.init_size, self.init_size)
#         img = self.conv_blocks0(out)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks1(img)
#         img = nn.functional.interpolate(img,scale_factor=2)
#         img = self.conv_blocks2(img)
#         return img

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

# class GeneratorC_mult(nn.Module):
#     '''
#     Conditional Generator
#     '''
#     def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
#         super(GeneratorC_mult, self).__init__()
#         self.num_classes = num_classes
#         self.generator_list = []
#         for i in range(num_classes):
#             self.generator_list.append(GeneratorC_single(nz, ngf, nc, img_size))
#             self.add_module(f"gen_{i}", self.generator_list[i])
#             self.generator_list[i].apply(init_weights)

#     def forward(self, z, label):
#         img_list = []
#         for i in range(label.size(0)):
#             img_list.append(self.generator_list[label[i].item()](z))
#         img = torch.stack(img_list)
#         del img_list
#         print(img.size())
#         return img


class GeneratorC_mult(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_generator = 10):
        super(GeneratorC_mult, self).__init__()
        self.num_classes = num_classes
        self.generator_list = []
        self.num_generator = num_generator
        self.count = 0
        for i in range(self.num_generator):

            # self.generator_list.append(get_decoder("res_normN4C64", output_nc = 3, input_dim = nz))
            self.generator_list.append(GeneratorC(nz, num_classes, ngf, nc, img_size))
            self.add_module(f"gen_{i}", self.generator_list[i])
            self.generator_list[i].apply(init_weights)

    def forward(self, z, label):
        # print(self.count % 10)
        img = self.generator_list[self.count % self.num_generator](z, label)
        self.count += 1
        if self.count == self.num_generator:
            self.count = 0
        return img


class GeneratorD(nn.Module):
    '''
    Unconditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(GeneratorD, self).__init__()
        
        # self.label_emb = nn.Embedding(nz)
        
        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        # label_inp = self.label_emb(label)
        # gen_input = torch.cat((label_inp, z), -1)
        gen_input = z
        out = self.l1(gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorD_mult(nn.Module):
    '''
    Un-conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_generator = 10):
        super(GeneratorD_mult, self).__init__()
        self.generator_list = []
        self.num_generator = num_generator
        self.count = 0
        for i in range(self.num_generator):
            self.generator_list.append(GeneratorD(nz, num_classes, ngf, nc, img_size))
            self.add_module(f"gen_{i}", self.generator_list[i])
            self.generator_list[i].apply(init_weights)

    def forward(self, z, label):
        # print(self.count % 10)
        img = self.generator_list[self.count % self.num_generator](z, label)
        self.count += 1
        if self.count == self.num_generator:
            self.count = 0
        return img


class Generator_resC(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(Generator_resC, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            ResBlock(ngf*2, ngf*2, bn = True, stride=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            ResBlock(ngf*2, ngf*2, bn = True, stride=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks3(img)
        return img

class Generator_resC_mult(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_generator = 10):
        super(Generator_resC_mult, self).__init__()
        self.num_classes = num_classes
        self.generator_list = []
        self.num_generator = num_generator
        self.count = 0
        for i in range(self.num_generator):

            # self.generator_list.append(get_decoder("res_normN4C64", output_nc = 3, input_dim = nz))
            self.generator_list.append(Generator_resC(nz, num_classes, ngf, nc, img_size))
            self.add_module(f"gen_{i}", self.generator_list[i])
            self.generator_list[i].apply(init_weights)

    def forward(self, z, label):
        # print(self.count % 10)
        img = self.generator_list[self.count % self.num_generator](z, label)
        self.count += 1
        if self.count == self.num_generator:
            self.count = 0
        return img

class GeneratorDynamic_A(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_heads = 10):
        super(GeneratorDynamic_A, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        
        self.num_heads = num_heads

        self.l1_list = []
        for i in range(num_heads):
            self.l1_list.append(nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2)))
            self.add_module(f"l1_{i}", self.l1_list[i])
            self.l1_list[i].apply(init_weights)

        self.conv_blocks_0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
        )

        self.count = 0

        self.conv_blocks1 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1_list[self.count](gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)

        img = self.conv_blocks_0(out)
        
        self.count += 1
        if self.count == self.num_heads:
            self.count = 0
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

class GeneratorDynamic_B(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_heads = 10):
        super(GeneratorDynamic_B, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        
        self.num_heads = num_heads

        self.l1 = nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2))

        
        self.conv_blocks_0_list = []
        for i in range(num_heads):
            self.conv_blocks_0_list.append(nn.Sequential(
                nn.BatchNorm2d(ngf*2),
                nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            ))
            self.add_module(f"gen_{i}", self.conv_blocks_0_list[i])
            self.conv_blocks_0_list[i].apply(init_weights)
        
        self.count = 0

        self.conv_blocks1 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)

        img = self.conv_blocks_0_list[self.count](out)
        
        self.count += 1
        if self.count == self.num_heads:
            self.count = 0
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

class GeneratorDynamic_C(nn.Module):
    '''
    Conditional Generator
    '''
    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32, num_heads = 10):
        super(GeneratorDynamic_C, self).__init__()
        
        self.label_emb = nn.Embedding(num_classes, nz)
        
        self.init_size = img_size//4
        
        self.num_heads = num_heads

        self.l1_list = []
        for i in range(num_heads):
            self.l1_list.append(nn.Sequential(nn.Linear(nz*2, ngf*2*self.init_size**2)))
            self.add_module(f"l1_{i}", self.l1_list[i])
            self.l1_list[i].apply(init_weights)

        
        self.conv_blocks_0_list = []
        for i in range(num_heads):
            self.conv_blocks_0_list.append(nn.Sequential(
                nn.BatchNorm2d(ngf*2),
                nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            ))
            self.add_module(f"gen_{i}", self.conv_blocks_0_list[i])
            self.conv_blocks_0_list[i].apply(init_weights)
        
        self.count = 0

        self.conv_blocks1 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1_list[self.count](gen_input.view(gen_input.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)

        img = self.conv_blocks_0_list[self.count](out)
        
        self.count += 1
        if self.count == self.num_heads:
            self.count = 0
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img













class ResTransposeBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=2):
        super(ResTransposeBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, output_padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_planes, planes, kernel_size=1, stride=stride, output_padding=1),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


def resnet(input_shape, level):
    # print(level)
    net = []

    net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
    net += [nn.BatchNorm2d(64)]
    net += [nn.ReLU()]
    net += [nn.MaxPool2d(2)]
    net += [ResBlock(64, 64)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(64, 128, stride=2)]

    if level == 2:
        return nn.Sequential(*net)
    
    net += [ResBlock(128, 128)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock(128, 256, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)

def resnet_tail(level, num_class = 10):
    print(level)
    net = []
    if level <= 1:
        net += [ResBlock(64, 128, stride=2)]
    if level <= 2:
        net += [ResBlock(128, 128)]
    if level <= 3:
        net += [ResBlock(128, 256, stride=2)]
    net += [ResBlock(256, 256, stride=1)]
    net += [ResBlock(256, 512, stride=2)]
    net += [ResBlock(512, 512, stride=1)]
    net += [ResBlock(512, 1024, stride=2)]
    net += [ResBlock(1024, 1024, stride=1)]
    # net += [nn.AvgPool2d(2)]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(num_class)]
    return nn.Sequential(*net)


def pilot(input_shape, level):

    net = []

    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    
    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)
    
    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def make_generator(latent_size):

    net = []
    net += [torch.nn.Linear(latent_size, 8*8*256, bias = False)]
    net += [torch.nn.BatchNorm1d(8*8*256)]
    net += [torch.nn.LeakyReLU()]
    net += [View((-1, 256, 8, 8))]
    net += [torch.nn.ConvTranspose2d(256, 128, 3, 1, padding = 1, bias = False)]
    net += [torch.nn.BatchNorm2d(128)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(128, 64, 3, 2, padding = 1, output_padding=1, bias = False)]
    net += [torch.nn.BatchNorm2d(64)]
    net += [torch.nn.LeakyReLU()]

    net += [torch.nn.ConvTranspose2d(64, 3, 3, 2, padding = 1, output_padding=1, bias = False)]
    net += [torch.nn.Tanh()]
    # net += [torch.nn.Sigmoid()]

    return nn.Sequential(*net)


def multihead_buffer(feature_size):
    assert len(feature_size) == 4
    net = []
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    net += [torch.nn.Conv2d(feature_size[1], feature_size[1], 3, 1, padding=1)]
    net += [torch.nn.BatchNorm2d(feature_size[1])]
    net += [torch.nn.ReLU()]
    return nn.Sequential(*net)

def multihead_buffer_res(feature_size):
    assert len(feature_size) == 4
    net = []
    net += [ResBlock(feature_size[1], feature_size[1])]
    net += [ResBlock(feature_size[1], feature_size[1])]
    # net += [ResBlock(feature_size[1], feature_size[1])]
    return nn.Sequential(*net)

def cifar_pilot(output_dim, level):

    net = []

    act = None
    #act = 'swish'
    
    print("[PILOT] activation: ", act)
    print(output_dim)
    if output_dim[2] == 32:
        net += [nn.Conv2d(3, 64, 3, 1, 1)]
        return  nn.Sequential(*net)

    net += [nn.Conv2d(3, 64, 3, 2, 1)]

    net += [nn.Conv2d(64, 64, 3, 1, 1)]

    if output_dim[2] == 16:
        net += [nn.Conv2d(64, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)


    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    net += [nn.Conv2d(128, 128, 3, 1, 1)]

    if output_dim[2] == 8:
        net += [nn.Conv2d(128, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)
    
    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if output_dim[2] == 4:
        net += [nn.Conv2d(256, output_dim[1], 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)
        

def decoder(input_shape, level, channels=3):
    
    net = []

    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)

    net += [nn.ConvTranspose2d(input_shape[0], 256, 3, 2, 1, output_padding=1)]

    if level == 1:
        net += [nn.Conv2d(256, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]

    if level <= 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    net += [nn.ConvTranspose2d(128, channels, 3, 2, 1, output_padding=1)]
    net += [nn.Tanh()]
    return nn.Sequential(*net)

def cifar_decoder(input_shape, channels=3):
    
    net = []

    #act = "relu"
    act = None
    
    print("[DECODER] activation: ", act)
    # print(input_shape)

    if input_shape[2] == 16:
        net += [nn.Conv2d(input_shape[0], 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    
    elif input_shape[2] == 8:
        net += [nn.Conv2d(input_shape[0], 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    elif input_shape[2] == 4:
        net += [nn.Conv2d(input_shape[0], 256, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        if act == "relu":
            net += [nn.ReLU()]
        net += [nn.ConvTranspose2d(64, channels, 3, 2, 1, output_padding=1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)
    else:
        raise Exception('No Dim %d' % input_shape[2])

# def inference_model(input_shape):
#     pass

class inference_model(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(inference_model, self).__init__()
        self.features=nn.Sequential(
            nn.Linear(num_classes,1024),
            # nn.Linear(num_classes,256),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            # nn.Linear(256,128),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,128),
            # nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            )
        self.labels=nn.Sequential(
           nn.Linear(num_classes,1024),
           nn.ReLU(),
            nn.Linear(1024,512),
        #    nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(512,128),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            )
        self.combine=nn.Sequential(
            nn.Linear(128*2,256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,1),
            )
        # for key in self.state_dict():
        #     if key.split('.')[-1] == 'weight':    
        #         nn.init.normal(self.state_dict()[key], std=0.01)
                
        #     elif key.split('.')[-1] == 'bias':
        #         self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    def forward(self,x,l):
        
        out_x = self.features(x)
        out_l = self.labels(l)
            
        is_member =self.combine( torch.cat((out_x  ,out_l),1))
    
        return self.output(is_member)


def discriminator(input_shape, bn = True, resblock_repeat = 2, dropout = False):
    # input is 32x32x3
    net = []
    net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
    # net += [nn.ReLU()]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    # net += [nn.ReLU()]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
        
    for _ in range(resblock_repeat):
        net += [ResBlock(256, 256, bn=bn)]
        if dropout:
            net += [nn.Dropout(0.2)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(1)]
    net += [nn.Sigmoid()]
    return nn.Sequential(*net)

def D2GAN_discriminator(input_shape, bn = True, resblock_repeat = 2, dropout = False):
    # input is 32x32x3
    net = []
    net += [nn.Conv2d(input_shape[0], 128, 3, 2, 1)]
    # net += [nn.ReLU()]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    # net += [nn.ReLU()]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
        
    for _ in range(resblock_repeat):
        net += [ResBlock(256, 256, bn=bn)]
        if dropout:
            net += [nn.Dropout(0.2)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    if dropout:
        net += [nn.Dropout(0.2)]
    net += [nn.ReLU()]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(1)]
    net += [nn.Softplus()]
    return nn.Sequential(*net)
#==========================================================================================

class custom_AE_bn(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE_bn, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(int(nc/2))]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(int(nc/2))]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(int(input_nc/(2 ** (upsampling_num - 1))))]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.BatchNorm2d(input_nc)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class custom_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(custom_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            #TODO: change to Conv2d
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class complex_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class complex_res_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_res_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            
            model += [ResBlock(nc, int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(int(nc/2), int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [ResBlock(input_nc, input_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(input_nc, output_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class conv_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(conv_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        model += [nn.Conv2d(input_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #first
        model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        for _ in range(N):
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #Middle-N
            model += [nn.BatchNorm2d(internal_nc)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [nn.Conv2d(internal_nc, internal_nc, kernel_size=3, stride=1, padding=1)] #two required
            model += [nn.BatchNorm2d(internal_nc)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [nn.Conv2d(internal_nc, output_nc, kernel_size=3, stride=1, padding=1)] #last
        model += [nn.BatchNorm2d(output_nc)]

        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output





class res_normN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(res_normN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
            
        model += [ResBlock(input_nc, internal_nc, bn = True, stride=1)] #first
        model += [nn.ReLU()]

        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
            model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)] #second
        model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(internal_nc)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
        model += [nn.ReLU()]

        if upsampling_num >= 3:
            for _ in range(upsampling_num - 2):
                model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
                model += [nn.BatchNorm2d(internal_nc)]
                model += [nn.ReLU()]

        model += [ResBlock(internal_nc, output_nc, bn = True, stride=1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output


class complex_normplusN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_normplusN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
            
        model += [ResBlock(input_nc, internal_nc, bn = True, stride=1)] #first
        model += [nn.ReLU()]
        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)] #second
        model += [nn.ReLU()]
        
        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
            model += [nn.ReLU()]

        if upsampling_num >= 2:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
        
        if upsampling_num >= 3:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
        
        if upsampling_num >= 4:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]

        if upsampling_num >= 5:
            model += [nn.ConvTranspose2d(internal_nc, internal_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
        else:
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]

        model += [nn.ReLU()]
        model += [ResBlock(internal_nc, output_nc, bn = True, stride=1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output

class TB_normplusN_AE(nn.Module):
    def __init__(self, N = 0, internal_nc = 64, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(TB_normplusN_AE, self).__init__()
        if input_dim != 0:
            upsampling_num = int(np.log2(output_dim // input_dim)) # input_dim =0 denotes confidensce score
            self.confidence_score = False
        else:
            upsampling_num = int(np.log2(output_dim))
            self.confidence_score = True
        model = []
        model += [nn.Conv2d(input_nc, internal_nc, stride = 1, kernel_size=3, padding= 1)]
        model += [nn.ReLU()]

        if upsampling_num >= 1:
            model += [ResTransposeBlock(internal_nc, internal_nc, bn = True, stride=2)]
            model += [nn.ReLU()]
        
        if upsampling_num >= 2:
            model += [ResTransposeBlock(internal_nc, internal_nc, bn = True, stride=2)]
            model += [nn.ReLU()]
        
        for _ in range(N):
            model += [ResBlock(internal_nc, internal_nc, bn = True, stride=1)]
            model += [nn.ReLU()]

        if upsampling_num >= 3:
            model += [ResTransposeBlock(internal_nc, internal_nc, bn = True, stride=2)]
            model += [nn.ReLU()]
        
        if upsampling_num >= 4:
            model += [ResTransposeBlock(internal_nc, internal_nc, bn = True, stride=2)]
            model += [nn.ReLU()]

        if upsampling_num >= 5:
            model += [ResTransposeBlock(internal_nc, internal_nc, bn = True, stride=2)]
            model += [nn.ReLU()]
        
        model += [nn.Conv2d(internal_nc, output_nc, stride = 1, kernel_size=3, padding= 1)]
        if activation == "sigmoid":
            model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        if self.confidence_score:
            x = x.view(x.size(0), x.size(2), 1, 1)
        output = self.m(x)
        return output

#Specialized Decoder for BottleNecked Arch:
# class complex_res_AE(nn.Module):
#     def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
#         super(complex_res_AE, self).__init__()
#         upsampling_num = int(np.log2(output_dim // input_dim))
#         # get [b, 3, 8, 8]
#         model = []
#         nc = input_nc
#         for num in range(upsampling_num - 1):
#             input_nc = 32
#             model += [ResBlock(nc, int(input_nc/2), bn = True, stride=1)]
#             model += [nn.ReLU()]
#             model += [nn.ConvTranspose2d(int(input_nc/2), int(input_nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
#             model += [nn.ReLU()]
#             model += [ResBlock(int(input_nc/2), int(input_nc/2), bn = True, stride=1)]
#             model += [nn.ReLU()]
#             input_nc = int(input_nc/2)
#         if upsampling_num >= 1:
#             model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), 16, bn = True, stride=1)]
#             model += [nn.ReLU()]
#             model += [nn.ConvTranspose2d(16, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
#             model += [nn.ReLU()]
#             model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
#             if activation == "sigmoid":
#                 model += [nn.Sigmoid()]
#             elif activation == "tanh":
#                 model += [nn.Tanh()]
#         else:
#             model += [ResBlock(input_nc, 16, bn = True, stride=1)]
#             model += [nn.ReLU()]
#             model += [ResBlock(16, output_nc, bn = True, stride=1)]
#             model += [nn.ReLU()]
#             model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
#             if activation == "sigmoid":
#                 model += [nn.Sigmoid()]
#             elif activation == "tanh":
#                 model += [nn.Tanh()]
#         self.m = nn.Sequential(*model)

#     def forward(self, x):
#         output = self.m(x)
#         return output


class complex_resplusN_AE(nn.Module):
    def __init__(self, N = 2, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_resplusN_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            for _ in range(N):
                model += [ResBlock(nc, int(nc), bn = True, stride=1)]
                model += [nn.ReLU()]
            model += [ResBlock(nc, int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(int(nc/2), int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            for _ in range(N):
                model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), bn = True, stride=1)]
                model += [nn.ReLU()]
            model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            for _ in range(N):
                model += [ResBlock(input_nc, input_nc, bn = True, stride=1)]
                model += [nn.ReLU()]
            model += [ResBlock(input_nc, input_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(input_nc, output_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class complex_resplus_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_resplus_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            
            model += [ResBlock(nc, int(nc), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(nc, int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(int(nc/2), int(nc/2), bn = True, stride=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [ResBlock(input_nc, input_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(input_nc, input_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(input_nc, output_nc, bn = True, stride=1)]
            model += [nn.ReLU()]
            model += [ResBlock(output_nc, output_nc, bn = True, stride=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class complex_plus_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(complex_plus_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            
            model += [nn.Conv2d(nc, int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(nc/2), int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(int(nc/2), int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(int(nc/2), int(nc/2), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.Conv2d(int(input_nc/(2 ** (upsampling_num - 1))), int(input_nc/(2 ** (upsampling_num - 1))), kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class simple_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(simple_AE, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.ConvTranspose2d(nc, int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output

class simple_AE_bn(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(simple_AE_bn, self).__init__()
        upsampling_num = int(np.log2(output_dim // input_dim))
        # get [b, 3, 8, 8]
        model = []
        nc = input_nc
        for num in range(upsampling_num - 1):
            model += [nn.ConvTranspose2d(nc, int(nc/2), kernel_size=3, stride=2, padding=1, output_padding=1)]
            model += [nn.BatchNorm2d(int(nc/2))]
            model += [nn.ReLU()]
            nc = int(nc/2)
        if upsampling_num >= 1:
            model += [nn.ConvTranspose2d(int(input_nc/(2 ** (upsampling_num - 1))), output_nc, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        else:
            model += [nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)]
            if activation == "sigmoid":
                model += [nn.Sigmoid()]
            elif activation == "tanh":
                model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        output = self.m(x)
        return output


class simplest_AE(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, input_dim=8, output_dim=32, activation = "sigmoid"):
        super(simplest_AE, self).__init__()
        self.output_dim = output_dim
        self.output_nc = output_nc
        stride = output_dim // input_dim
        kernel_size = ((stride + 1) // 2) * 2 + 1
        print("stride size: {}, kernel size: {}".format(stride, kernel_size))

        if stride == 1:
            output_padding_size = 0
        elif stride == 2:
            output_padding_size = 1
        elif stride == 4:
            output_padding_size = 3
        elif stride == 8:
            output_padding_size = 5
        else:
            output_padding_size = 0
        
        model = []
        # model += [nn.Linear(input_nc * input_dim * input_dim, output_nc * output_dim * output_dim)]
        model += [nn.ConvTranspose2d(input_nc, output_nc, kernel_size= kernel_size, stride=stride, padding=kernel_size // 2, output_padding = output_padding_size)]
        if activation == "sigmoid":
                model += [nn.Sigmoid()]
        elif activation == "tanh":
            model += [nn.Tanh()]
        self.m = nn.Sequential(*model)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        output = self.m(x)
        # output = output.view(output.size(0), self.output_nc, self.output_dim, self.output_dim)
        return output

def classifier_binary(input_shape, class_num):
    net = []
    # xin = tf.keras.layers.Input(input_shape)
    # net += [nn.ReLU()]
    # net += [nn.Conv2d(input_shape[0], 128, 3, 1, 1)]
    # net += [nn.ReLU()]
    # net += [ResBlock(128, 128, bn=True)]
    # net += [ResBlock(128, 128, bn=True)]
    net += [nn.ReLU()]
    net += [nn.Flatten()]
    net += [nn.LazyLinear(256)]
    net += [nn.ReLU()]
    net += [nn.Linear(256, 128)]
    net += [nn.ReLU()]
    # if(class_num > 1):
    #     net += [nn.BatchNorm2d(np.prod([input_shape[0], 32, input_shape[2], input_shape[3]]))]
    net += [nn.Linear(128, class_num)]
    return nn.Sequential(*net)

def pilotClass(input_shape, level):
    net = []
    # xin = tf.keras.layers.Input(input_shape)

    net += [nn.Conv2d(input_shape[0], 64, 3, 2, 1)]
    net += [nn.SiLU]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]
    net += [nn.SiLU]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]
    net += [nn.SiLU]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)
        
SETUPS = [(functools.partial(resnet, level=i), functools.partial(pilot, level=i), functools.partial(decoder, level=i), functools.partial(discriminator, level=i), functools.partial(resnet_tail, level=i)) for i in range(1,6)]

# bin class
l = 4
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l), functools.partial(resnet_tail, level=l))]

l = 3
SETUPS += [(functools.partial(resnet, level=l), functools.partial(pilot, level=l), classifier_binary, functools.partial(discriminator, level=l), functools.partial(resnet_tail, level=l))]



# if __name__ == "__main__":
#     model1 = GeneratorC(nz = 512, ngf=128, nc=3)
#     print(model1)