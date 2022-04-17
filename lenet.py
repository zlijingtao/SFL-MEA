'''LeNet5 in PyTorch.
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


class LeNet(nn.Module):
    '''
    LeNet model 
    '''
    def __init__(self, feature, logger, num_client = 1, num_class = 10, initialize_different = False):
        super(LeNet, self).__init__()
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
                    

        self.local = self.local_list[0]
        self.cloud = feature[1]
        self.logger = logger
        classifier_list = [nn.Linear(256, 120), nn.ReLU(True),nn.Linear(120, 84), nn.ReLU(True)]
        classifier_list += [nn.Linear(84, num_class)]
        self.classifier = nn.Sequential(*classifier_list)
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
            noise_input = torch.randn([1, 1, 28, 28])
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
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cutting_layer):
    real_cutlayer = cutting_layer
    if cutting_layer == 1:
        real_cutlayer = 3
    elif cutting_layer >= 2:
        real_cutlayer = 6
    layers = [          
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            ]
    try:
        local_layer_list = layers[:real_cutlayer]
        cloud_layer_list = layers[real_cutlayer:]
    except:
        print("Cutting layer is greater than overall length of the ResNet arch! set cloud to empty list")
        local_layer_list = layers[:]
        cloud_layer_list = []

    local = nn.Sequential(*local_layer_list)
    cloud = nn.Sequential(*cloud_layer_list)
    return local, cloud

def LeNet5(cutting_layer, logger, num_client = 1, num_class = 10, initialize_different = False):
    return LeNet(make_layers(cutting_layer), logger, num_client = num_client, num_class = num_class, initialize_different = initialize_different)

def test():
    net = LeNet5(1, None)
    y = net(torch.randn(1, 1, 28, 28))
    print(y.size())

test()
