from architectures_torch import *
from vgg import vgg11_bn
from resnet_cifar import ResNet20
from torchsummary import summary

'''Select Model below'''

# resnet_cut3 = ResNet20(3, None, num_class = 10)
# resnet_cut3.cuda()
# smashed_data_size = resnet_cut3.get_smashed_data_size()


# vgg_cut3 = vgg11_bn(2, None, num_class = 10)
vgg_cut3 = ResNet20(2, None, num_class = 10)
vgg_cut3.cuda()
smashed_data_size = vgg_cut3.get_smashed_data_size()

print(vgg_cut3)
'''Select AE model below'''

print("Smashed data size is", smashed_data_size)
#Tier0, 1, 2, 3, 4 Decoderr Architecture
# AE_model = complex_resplus_AE(input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()
AE_model = complex_res_AE(input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()
# AE_model = complex_AE(input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()
# AE_model = custom_AE(input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()
# AE_model = simple_AE(input_nc=smashed_data_size[1], input_dim=smashed_data_size[2]).cuda()


'''Print out Model Summary'''
print(AE_model)
# summary(AE_model, smashed_data_size[1:])