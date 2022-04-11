'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None: 
            m.bias.data.zero_()

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class conv3x3(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride = stride, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return out



class ResNet(nn.Module):
    '''
    ResNet model 
    '''
    def __init__(self, feature, logger, expansion = 1, num_client = 1, num_class = 10, initialize_different = False):
        super(ResNet, self).__init__()
        self.current_client = 0
        self.local_list = []
        for i in range(num_client):
            if i == 0:
                self.local_list.append(feature[0])
                self.local_list[0].apply(init_weights)
            else:
                new_copy = copy.deepcopy(self.local_list[0])

                self.local_list.append(new_copy.cuda())
                if initialize_different:
                    self.local_list[i].apply(init_weights)
                    
            # for name, params in self.local_list[-1].named_parameters():
            #     print(name, 'of client', i, params.data[1][1])
            #     break

        self.local = self.local_list[0]
        self.cloud = feature[1]
        self.image_size = feature[2]
        self.logger = logger
        self.classifier = nn.Linear(512*expansion, num_class)
        print("local:")
        print(self.local)
        print("cloud:")
        print(self.cloud)
        print("classifier:")
        print(self.classifier)
         # Initialize weights
        for m in self.cloud:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    def switch_model(self, client_id):
        self.current_client = client_id
        self.local = self.local_list[client_id]

    def get_current_client(self):
        return self.current_client

    def get_smashed_data_size(self):
        with torch.no_grad():
            noise_input = torch.randn([1, 3, 224, 224])
            try:
                device = next(self.local.parameters()).device
                noise_input = noise_input.to(device)
            except:
                pass
            smashed_data = self.local(noise_input)
        return smashed_data.size()
    
    def forward(self, x):
        self.local_output = self.local(x)
        x = self.cloud(self.local_output)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(block, layer_list, cutting_layer, adds_bottleneck = False, bottleneck_option = "C8S1"):

    layers = []
    current_image_dim = 32
    count = 1
    layers.append(conv3x3(3, 64))
    in_planes = 64

    strides = [1] + [1]*(layer_list[0]-1)
    for stride in strides:
        layers.append(block(in_planes, 64, stride))
        count += 1
        current_image_dim = current_image_dim // stride

        in_planes = 64 * block.expansion

    strides = [2] + [1]*(layer_list[1]-1)
    for stride in strides:
        layers.append(block(in_planes, 128, stride))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 128 * block.expansion

    strides = [2] + [1]*(layer_list[2]-1)
    for stride in strides:
        layers.append(block(in_planes, 256, stride))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 256 * block.expansion

    strides = [2] + [1]*(layer_list[3]-1)
    for stride in strides:
        layers.append(block(in_planes, 512, stride))
        count += 1
        current_image_dim = current_image_dim // stride
        in_planes = 512 * block.expansion
    try:
        local_layer_list = layers[:cutting_layer]
        cloud_layer_list = layers[cutting_layer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []

    temp_local = nn.Sequential(*local_layer_list)
    with torch.no_grad():
        
        noise_input = torch.randn([1, 3, 32, 32])
        smashed_data = temp_local(noise_input)
    input_nc = smashed_data.size(1)
    
    print("in_channels is {}".format(input_nc))

    local = []
    cloud = []
    if adds_bottleneck: # to enable gooseneck, simply copy below to other architecture
        print("original channel size of smashed-data is {}".format(input_nc))
        try:
            if "noRELU" in bottleneck_option or "norelu" in bottleneck_option or "noReLU" in bottleneck_option:
                relu_option = False
            else:
                relu_option = True
            if "K" in bottleneck_option:
                bn_kernel_size = int(bottleneck_option.split("C")[0].split("K")[1])
            else:
                bn_kernel_size = 3
            bottleneck_channel_size = int(bottleneck_option.split("S")[0].split("C")[1])
            if "S" in bottleneck_option:
                bottleneck_stride = int(bottleneck_option.split("S")[1])
            else:
                bottleneck_stride = 1
        except:
            print("auto extract bottleneck option fail (format: CxSy, x = [1, max_channel], y = {1, 2}), set channel size to 8 and stride to 1")
            bn_kernel_size = 3
            bottleneck_channel_size = 8
            bottleneck_stride = 1
            relu_option = True
        # cleint-side bottleneck
        if bottleneck_stride == 1:
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            local += [nn.Conv2d(input_nc, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
            for _ in range(int(np.log2(bottleneck_stride//2))):
                if relu_option:
                    local += [nn.ReLU()]
                local += [nn.Conv2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, padding=1, stride= 2)]
        if relu_option:
            local += [nn.ReLU()]
        # server-side bottleneck
        if bottleneck_stride == 1:
            cloud += [nn.Conv2d(bottleneck_channel_size, input_nc, kernel_size=bn_kernel_size, padding=bn_kernel_size//2, stride= 1)]
        elif bottleneck_stride >= 2:
            for _ in range(int(np.log2(bottleneck_stride//2))):
                cloud += [nn.ConvTranspose2d(bottleneck_channel_size, bottleneck_channel_size, kernel_size=3, output_padding=1, padding=1, stride= 2)]
                if relu_option:
                    cloud += [nn.ReLU()]
            cloud += [nn.ConvTranspose2d(bottleneck_channel_size, input_nc, kernel_size=3, output_padding=1, padding=1, stride= 2)]
        if relu_option:
            cloud += [nn.ReLU()]
        print("added bottleneck, new channel size of smashed-data is {}".format(bottleneck_channel_size))
        input_nc = bottleneck_channel_size
    local_layer_list += local
    cloud_layer_list = cloud + cloud_layer_list
    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)

    print("image size of cutting layer is [-1, {}, {}, {}]".format(input_nc, current_image_dim, current_image_dim))
    image_size = (input_nc, current_image_dim, current_image_dim)
    return local, cloud, image_size

def ResNet18(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return ResNet(make_layers(BasicBlock, [2, 2, 2, 2], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different)


def ResNet34(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return ResNet(make_layers(BasicBlock, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different)


def ResNet50(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return ResNet(make_layers(Bottleneck, [3, 4, 6, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, expansion= 4, num_client = num_client, num_class = num_class, initialize_different = initialize_different)


def ResNet101(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return ResNet(make_layers(Bottleneck, [3, 4, 23, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, expansion= 4, num_client = num_client, num_class = num_class, initialize_different = initialize_different)


def ResNet152(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False, adds_bottleneck = False, bottleneck_option = "C8S1"):
    return ResNet(make_layers(Bottleneck, [3, 8, 36, 3], cutting_layer, adds_bottleneck = adds_bottleneck, bottleneck_option = bottleneck_option), logger, expansion= 4, num_client = num_client, num_class = num_class, initialize_different = initialize_different)


# def test():
#     net = ResNet18(1)
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())

# test()
