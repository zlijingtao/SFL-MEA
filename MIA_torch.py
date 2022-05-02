from dis import dis
from re import T
from selectors import EpollSelector
import pickle
import torch
import numpy as np
import torch.nn as nn
from torch.serialization import save
import torchvision.transforms as transforms
import architectures_torch as architectures
from utils import setup_logger, accuracy, fidelity, set_bn_eval, student_Loss, AverageMeter, WarmUpLR, apply_transform_test, apply_transform, TV, l2loss, dist_corr, get_PSNR, zeroing_grad
from utils import freeze_model_bn, average_weights, DistanceCorrelationLoss, spurious_loss, prune_top_n_percent_left, dropout_defense, prune_defense
from thop import profile
import logging
from torch.autograd import Variable
from resnet import ResNet18, ResNet34
from resnet_cifar import ResNet20, ResNet32
from mobilenetv2 import MobileNetV2
from vgg import vgg11, vgg13, vgg11_bn, vgg13_bn
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from datetime import datetime
import os, copy
from shutil import rmtree
from datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_imagenet_trainloader, get_imagenet_testloader, get_mnist_bothloader, get_facescrub_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_tinyimagenet_bothloader
from tqdm import tqdm
import glob
DENORMALIZE_OPTION=True
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "mnist" or dataset == "fmnist":
        return torch.clamp((x + 1)/2, 0, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "tinyimagenet":
        mean = (0.5141, 0.5775, 0.3985)
        std = (0.2927, 0.2570, 0.1434)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = tensor[t].mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)

def save_images(input_imgs, output_imgs, epoch, path, offset=0, batch_size=64):
    """
    """
    input_prefix = "inp_"
    output_prefix = "out_"
    out_folder = "{}/{}".format(path, epoch)
    out_folder = os.path.abspath(out_folder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    for img_idx in range(output_imgs.shape[0]):
        inp_img_path = "{}/{}{}.jpg".format(out_folder, input_prefix, offset * batch_size + img_idx)
        out_img_path = "{}/{}{}.jpg".format(out_folder, output_prefix, offset * batch_size + img_idx)

        if input_imgs is not None:
            save_image(input_imgs[img_idx], inp_img_path)
        if output_imgs is not None:
            save_image(output_imgs[img_idx], out_img_path)


class MIA:
    def __init__(self, arch, cutting_layer, batch_size, n_epochs, scheme="V3", num_client=2, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0,
                 collude_use_public=False, initialize_different=False, learning_rate=0.1, local_lr = -1,
                 collude_not_regularize=False, gan_AE_type="custom", random_seed=123,
                 num_client_regularize=1, pre_train_ganadv=False, load_from_checkpoint = False, bottleneck_option="None", measure_option=False,
                 optimize_computation=1, decoder_sync = False, bhtsne_option = False, gan_loss_type = "SSIM", attack_confidence_score = False,
                 ssim_threshold = 0.0, finetune_freeze_bn = False, load_from_checkpoint_server = False, source_task = "cifar100", client_sample_ratio = 1.0,
                 save_activation_tensor = False, save_more_checkpoints = False, dataset_portion = 1.0, noniid = 1.0, div_lambda = 0.0):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.bhtsne = bhtsne_option
        self.batch_size = batch_size
        self.lr = learning_rate
        self.finetune_freeze_bn = finetune_freeze_bn
        self.dataset_portion = dataset_portion
        self.client_sample_ratio = client_sample_ratio
        
        self.noniid_ratio = noniid
        if local_lr == -1: # if local_lr is not set
            self.local_lr = self.lr
        else:
            self.local_lr = local_lr
        self.n_epochs = n_epochs
        self.measure_option = measure_option
        self.optimize_computation = optimize_computation

        self.num_client_regularize = num_client
        self.divergence_aware = False
        if div_lambda != 0.0:
            self.divergence_aware = True

        self.div_lambda = div_lambda

        if collude_not_regularize and num_client_regularize <= num_client:
            self.num_client_regularize = num_client_regularize

        # setup save folder
        if save_dir is None:
            self.save_dir = "./saves/{}/".format(datetime.today().strftime('%m%d%H%M'))
        else:
            self.save_dir = str(save_dir) + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_more_checkpoints = save_more_checkpoints
        
        # setup tensorboard
        tensorboard_path = str(save_dir) + "/tensorboard"
        if not os.path.isdir(tensorboard_path):
            os.makedirs(tensorboard_path)
        self.writer = SummaryWriter(log_dir=tensorboard_path)
        self.save_activation_tensor = save_activation_tensor

        # setup logger
        model_log_file = self.save_dir + '/MIA.log'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger('{}_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
        self.warm = 1
        self.scheme = scheme

        # migrate old naming:
        if self.scheme == "V1" or self.scheme == "V2" or self.scheme == "V3" or self.scheme == "V4":
            self.scheme = self.scheme + "_batch"

        self.num_client = num_client
        self.dataset = dataset
        self.call_resume = False

        self.load_from_checkpoint = load_from_checkpoint
        self.load_from_checkpoint_server = load_from_checkpoint_server
        self.source_task = source_task
        # self.confidence_score = False
        self.cutting_layer = cutting_layer

        if self.cutting_layer == 0:
            self.logger.debug("Centralized Learning Scheme:")
        if "resnet20" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/10".format(self.cutting_layer))
        if "vgg11" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/13".format(self.cutting_layer))
        if "mobilenetv2" in arch:
            self.logger.debug("Split Learning Scheme: Overall Cutting_layer {}/9".format(self.cutting_layer))
        self.confidence_score = attack_confidence_score
        self.collude_use_public = collude_use_public
        self.initialize_different = initialize_different
        
        if "C" in bottleneck_option or "S" in bottleneck_option:
            self.adds_bottleneck = True
            self.bottleneck_option = bottleneck_option
        else:
            self.adds_bottleneck = False
            self.bottleneck_option = bottleneck_option
        
        self.decoder_sync = decoder_sync
        
        # Activation Defense:
        self.regularization_option = regularization_option

        # If strength is 0.0, then there is no regularization applied, train normally.
        self.regularization_strength = regularization_strength
        
        '''Train time ME attack'''
        self.trigger_stop = False


        self.gan_train_start_epoch = 0

        if "gan_train_ME" in self.regularization_option:
            self.Generator_train_option = True
            self.noise_w = 50.0

            if "nz256" in self.regularization_option:
                self.nz = 256
            elif "nz128" in self.regularization_option:
                self.nz = 128
            else:
                self.nz = 512
            try:
                self.gan_train_start_epoch = int(self.regularization_option.split("start")[1])
            except:
                self.logger.debug("extraing start epoch setting from arg, failed, set start_epoch to 160")
                self.gan_train_start_epoch = 160
            self.logger.debug(f"Perform gan_train_ME, start at {self.gan_train_start_epoch} epoch")
        else:
            self.Generator_train_option = False
        
        
        if "craft_train_ME" in self.regularization_option:
            self.Craft_train_option = True
            try:
                self.gan_train_start_epoch = int(self.regularization_option.split("start")[1])
            except:
                self.logger.debug("extraing start epoch setting from arg, failed, set start_epoch to 160")
                self.gan_train_start_epoch = 160
            
            self.craft_image_id = 0
            self.craft_step_count = 0
            self.num_craft_step = int(self.regularization_strength)
            if self.num_craft_step < 1:
                self.num_craft_step = 1
            self.logger.debug(f"Perform craft_train_ME, num_step to craft each image is {self.num_craft_step}, start at {self.gan_train_start_epoch} epoch")
        else:
            self.Craft_train_option = False
        
        if "GM_train_ME" in self.regularization_option:
            self.GM_train_option = True
            self.soft_image_id = 0
            self.soft_train_count = 0
            self.GM_data_proportion = self.regularization_strength # use reguarlization_strength to set data_proportion
            
            try:
                self.gan_train_start_epoch = int(self.regularization_option.split("start")[1])
            except:
                self.logger.debug("extraing start epoch setting from arg, failed, set start_epoch to 160")
                self.gan_train_start_epoch = 160
            self.logger.debug(f"Perform GM_train_ME, GM data proportion is {self.GM_data_proportion}, start at {self.gan_train_start_epoch} epoch")
        else:
            self.GM_train_option = False

        if "normal_train_ME" in self.regularization_option:
            
            self.Normal_train_option = True
            self.soft_image_id = 0
            self.soft_train_count = 0
            try:
                self.gan_train_start_epoch = int(self.regularization_option.split("start")[1])
            except:
                self.logger.debug("extraing start epoch setting from arg, failed, set start_epoch to 160")
                self.gan_train_start_epoch = 160
            
            self.logger.debug(f"Perform Normal_train_ME, start collecting gradient at {self.gan_train_start_epoch} epoch")
        else:
            self.Normal_train_option = False
        
        if "soft_train_ME" in self.regularization_option:
            self.Soft_train_option = True
            self.soft_image_id = 0
            self.soft_train_count = 0
            try:
                self.gan_train_start_epoch = int(self.regularization_option.split("start")[1])
            except:
                print("extraing start epoch setting from arg, failed, set start_epoch to 160")
                self.gan_train_start_epoch = 160
            self.logger.debug(f"Perform soft_train_ME, start at {self.gan_train_start_epoch} epoch")
        else:
            self.Soft_train_option = False

        if self.regularization_strength == 0.0:
            self.regularization_option = "None"
        

        # setup nopeek regularizer
        if "nopeek" in self.regularization_option:
            self.nopeek = True
        else:
            self.nopeek = False





        self.alpha1 = regularization_strength  # set to 0.1 # 1000 in Official NoteBook https://github.com/tremblerz/nopeek/blob/master/noPeekCifar10%20(1)-Copy2.ipynb

        # setup gan_adv regularizer
        self.gan_AE_activation = "sigmoid"
        self.gan_AE_type = gan_AE_type
        self.gan_loss_type = gan_loss_type
        self.gan_decay = 0.2
        self.alpha2 = regularization_strength  # set to 1~10
        self.pre_train_ganadv = pre_train_ganadv
        self.pretrain_epoch = 100

        self.ssim_threshold = ssim_threshold
        if "gan_adv" in self.regularization_option:
            self.gan_regularizer = True
            if "step" in self.regularization_option:
                try:
                    self.gan_num_step = int(self.regularization_option.split("step")[-1])
                except:
                    print("Auto extract step fail, geting default value 1")
                    self.gan_num_step = 1
            else:
                self.gan_num_step = 1
            if "noise" in self.regularization_option:
                self.gan_noise = True
            else:
                self.gan_noise = False
        else:
            self.gan_regularizer = False
            self.gan_noise = False
            self.gan_num_step = 1
        
        # setup local dp (noise-injection defense)
        if "local_dp" in self.regularization_option:
            self.local_DP = True
        else:
            self.local_DP = False

        self.dp_epsilon = regularization_strength

        if "dropout" in self.regularization_option:
            self.dropout_defense = True
            try: 
                self.dropout_ratio = float(self.regularization_option.split("dropout")[1].split("_")[0])
            except:
                self.dropout_ratio = regularization_strength
                print("Auto extract dropout ratio fail, use regularization_strength input as dropout ratio")
        else:
            self.dropout_defense = False
            self.dropout_ratio = regularization_strength
        
        if "topkprune" in self.regularization_option:
            self.topkprune = True
            try: 
                self.topkprune_ratio = float(self.regularization_option.split("topkprune")[1].split("_")[0])
            except:
                self.topkprune_ratio = regularization_strength
                print("Auto extract topkprune ratio fail, use regularization_strength input as topkprune ratio")
        else:
            self.topkprune = False
            self.topkprune_ratio = regularization_strength
        
        # dividing datasets to actual number of clients, self.num_clients is fake num of clients for ease of simulation.
        multiplier = 1/self.client_sample_ratio #100
        actual_num_users = int(multiplier * self.num_client)
        self.actual_num_users = actual_num_users


        if self.Generator_train_option or self.Craft_train_option or self.GM_train_option: 
            #data-free GAN-attack
            self.actual_num_users = self.actual_num_users - 1 # we let first N-1 client divide the training data, and skip the last client.

        # setup dataset
        if self.dataset == "cifar10":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar10_trainloader(batch_size=self.batch_size,
                                                                                                        num_workers=4,
                                                                                                        shuffle=True,
                                                                                                        num_client=actual_num_users,
                                                                                                        collude_use_public=self.collude_use_public,
                                                                                                         data_portion=self.dataset_portion, noniid_ratio = self.noniid_ratio)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar10_testloader(batch_size=self.batch_size,
                                                                                                        num_workers=4,
                                                                                                        shuffle=False)
            self.orig_class = 10
        elif self.dataset == "cifar100":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_cifar100_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public,
                                                                                                         data_portion=self.dataset_portion, noniid_ratio = self.noniid_ratio)
            # print()
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_cifar100_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=False)
            self.orig_class = 100

        elif self.dataset == "imagenet":
            self.client_dataloader = get_imagenet_trainloader(batch_size=self.batch_size,
                                                                num_workers=4,
                                                                shuffle=True,
                                                                num_client=actual_num_users,
                                                                collude_use_public=self.collude_use_public,
                                                                data_portion=self.dataset_portion, noniid_ratio = self.noniid_ratio)
            # print()
            self.pub_dataloader = get_imagenet_testloader(batch_size=self.batch_size,
                                                            num_workers=4,
                                                            shuffle=False)
            self.orig_class = 1000
        elif self.dataset == "svhn":
            self.client_dataloader, self.mem_trainloader, self.mem_testloader = get_SVHN_trainloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=True,
                                                                                                         num_client=actual_num_users,
                                                                                                         collude_use_public=self.collude_use_public,
                                                                                                         data_portion=self.dataset_portion)
            self.pub_dataloader, self.nomem_trainloader, self.nomem_testloader = get_SVHN_testloader(batch_size=self.batch_size,
                                                                                                         num_workers=4,
                                                                                                         shuffle=False)
            self.orig_class = 10

        elif self.dataset == "facescrub":
            self.client_dataloader, self.pub_dataloader = get_facescrub_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 530
        elif self.dataset == "tinyimagenet":
            self.client_dataloader, self.pub_dataloader = get_tinyimagenet_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 200
        elif self.dataset == "mnist":
            self.client_dataloader, self.pub_dataloader = get_mnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        elif self.dataset == "fmnist":
            self.client_dataloader, self.pub_dataloader = get_fmnist_bothloader(batch_size=self.batch_size, 
                                                                                num_workers=4,
                                                                                shuffle=True,
                                                                                num_client=actual_num_users,
                                                                                collude_use_public=self.collude_use_public)
            self.orig_class = 10
        else:
            raise ("Dataset {} is not supported!".format(self.dataset))
        self.num_class = self.orig_class

        if self.Generator_train_option or self.Craft_train_option or self.GM_train_option: 
            #data-free GAN-attack
            self.actual_num_users = self.actual_num_users + 1 # we let first N-1 client divide the training data, and skip the last client.

        self.num_batches = len(self.client_dataloader[0])
        print("Total number of batches per epoch for each client is ", self.num_batches)

        if self.Generator_train_option:
            # initialize generator.
            self.generator = architectures.GeneratorC(nz=self.nz, num_classes = self.num_class, ngf=128, nc=3, img_size=32)



        self.model = None

        if "V" in self.scheme:
            # V1, V2 initialize must be the same
            if "V1" in self.scheme or "V2" in self.scheme:
                self.initialize_different = False

            if arch == "resnet18":
                model = ResNet18(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet20":
                model = ResNet20(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet32":
                model = ResNet32(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet34":
                model = ResNet34(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13":
                model = vgg13(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11":
                model = vgg11(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13_bn":
                model = vgg13_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11_bn":
                model = vgg11_bn(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "mobilenetv2":
                model = MobileNetV2(cutting_layer, self.logger, num_client=self.num_client, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            else:
                raise ("No such architecture!")
            self.model = model

            self.f = model.local_list[0]
            if self.num_client > 1:
                self.c = model.local_list[1]
            self.classifier = model.classifier
            self.f.cuda()
            self.model.cloud.cuda()
            self.classifier.cuda()

            self.params = list(self.model.cloud.parameters()) + list(self.classifier.parameters())
            self.local_params = []
            if cutting_layer > 0:
                self.local_params.append(self.f.parameters())
                for i in range(1, self.num_client):
                    self.model.local_list[i].cuda()
                    self.local_params.append(self.model.local_list[i].parameters())

                if self.Generator_train_option:
                    self.generator.cuda()
                    self.local_params.append(self.generator.parameters())
                    
        else:
            # If not V3, we set num_client to 1 when initializing the model, because there is only one version of local model.
            if arch == "resnet18":
                model = ResNet18(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet20":
                model = ResNet20(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet32":
                model = ResNet32(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "resnet34":
                model = ResNet34(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13":
                model = vgg13(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11":
                model = vgg11(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                              initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg13_bn":
                model = vgg13_bn(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "vgg11_bn":
                model = vgg11_bn(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            elif arch == "mobilenetv2":
                model = MobileNetV2(cutting_layer, self.logger, num_client=1, num_class=self.num_class,
                                 initialize_different=self.initialize_different, adds_bottleneck=self.adds_bottleneck, bottleneck_option = self.bottleneck_option)
            else:
                raise ("No such architecture!")
            self.model = model
            self.f = model.local
            self.c = self.f
            for i in range(1, self.num_client):
                self.model.local_list.append(self.f)
            self.model.cloud = model.cloud
            self.classifier = model.classifier
            self.f.cuda()
            self.model.cloud.cuda()
            self.classifier.cuda()
            self.params = list(self.model.cloud.parameters()) + list(self.classifier.parameters())
            self.local_params = []
            if cutting_layer > 0:
                self.local_params.append(self.f.parameters())
        
        server_lr = self.lr

        # setup optimizers
        self.optimizer = torch.optim.SGD(self.params, lr=server_lr, momentum=0.9, weight_decay=5e-4)
        if self.pre_train_ganadv:
            milestones = [self.pretrain_epoch - 20, self.pretrain_epoch + 60, self.pretrain_epoch + 140]
        else:
            milestones = [60, 120, 160]
            if self.client_sample_ratio < 1.0:
                multiplier = 1/self.client_sample_ratio
                for i in range(len(milestones)):
                    milestones[i] = int(milestones[i] * multiplier)
        self.local_optimizer_list = []
        self.train_local_scheduler_list = []
        self.warmup_local_scheduler_list = []
        for i in range(len(self.local_params)):
            self.local_optimizer_list.append(torch.optim.SGD(list(self.local_params[i]), lr=self.local_lr, momentum=0.9, weight_decay=5e-4))
            self.train_local_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.local_optimizer_list[i], milestones=milestones,
                                                                    gamma=0.2))  # learning rate decay
            self.warmup_local_scheduler_list.append(WarmUpLR(self.local_optimizer_list[i], self.num_batches * self.warm))

        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.num_batches * self.warm)
        
        # Set up GAN_ADV
        self.local_AE_list = []
        self.gan_params = []
        if self.gan_regularizer:
            feature_size = self.model.get_smashed_data_size()
            for i in range(self.num_client):
                if self.gan_AE_type == "custom":
                    self.local_AE_list.append(
                        architectures.custom_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "custom_bn":
                    self.local_AE_list.append(
                        architectures.custom_AE_bn(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                   output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "complex":
                    self.local_AE_list.append(
                        architectures.complex_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                 output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "complex_plus":
                    self.local_AE_list.append(
                        architectures.complex_plus_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                      output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "complex_res":
                    self.local_AE_list.append(
                        architectures.complex_res_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                     output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "complex_resplus":
                    self.local_AE_list.append(architectures.complex_resplus_AE(input_nc=feature_size[1], output_nc=3,
                                                                               input_dim=feature_size[2], output_dim=32,
                                                                               activation=self.gan_AE_activation))
                elif "complex_resplusN" in self.gan_AE_type or self.gan_AE_type == "complex_resplus":
                    try:
                        N = int(self.gan_AE_type.split("complex_resplusN")[1])
                    except:
                        print("auto extract N from complex_resplusN failed, set N to default 2")
                        N = 2
                    self.local_AE_list.append(architectures.complex_resplusN_AE(N = N, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                elif "complex_normplusN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("complex_normplusN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from complex_normplusN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                
                elif "conv_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("conv_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from conv_normN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                
                elif "res_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("res_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from res_normN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                
                elif "TB_normplusN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("TB_normplusN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from TB_normplusN failed, set N to default 0")
                        N = 0
                        internal_C = 64
                    self.local_AE_list.append(architectures.TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=feature_size[1], output_nc=3,
                                                             input_dim=feature_size[2], output_dim=32,
                                                             activation=self.gan_AE_activation))
                
                elif self.gan_AE_type == "simple":
                    self.local_AE_list.append(
                        architectures.simple_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "simple_bn":
                    self.local_AE_list.append(
                        architectures.simple_AE_bn(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                   output_dim=32, activation=self.gan_AE_activation))
                elif self.gan_AE_type == "simplest":
                    self.local_AE_list.append(
                        architectures.simplest_AE(input_nc=feature_size[1], output_nc=3, input_dim=feature_size[2],
                                                  output_dim=32, activation=self.gan_AE_activation))
                else:
                    raise ("No such GAN AE type.")
                self.gan_params.append(self.local_AE_list[i].parameters())
                self.local_AE_list[i].apply(init_weights)
                self.local_AE_list[i].cuda()
            
            self.gan_optimizer_list = []
            self.gan_scheduler_list = []
            if self.pre_train_ganadv:
                milestones = [self.pretrain_epoch - 20, self.pretrain_epoch + 60, self.pretrain_epoch + 140]
            else:
                milestones = [60, 120, 160]
                if self.client_sample_ratio < 1.0:
                    multiplier = 1/self.client_sample_ratio
                    for i in range(len(milestones)):
                        milestones[i] = int(milestones[i] * multiplier)
            for i in range(len(self.gan_params)):
                self.gan_optimizer_list.append(torch.optim.Adam(list(self.gan_params[i]), lr=1e-3))
                self.gan_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.gan_optimizer_list[i], milestones=milestones,
                                                                      gamma=self.gan_decay))  # learning rate decay

    def optimizer_step(self, set_client = False, client_id = 0):
        self.optimizer.step()
        if set_client and len(self.local_optimizer_list) > client_id:
            self.local_optimizer_list[client_id].step()
        else:
            for i in range(len(self.local_optimizer_list)):
                self.local_optimizer_list[i].step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        for i in range(len(self.local_optimizer_list)):
            self.local_optimizer_list[i].zero_grad()

    def scheduler_step(self, epoch = 0, warmup = False):
        if warmup:
            self.warmup_scheduler.step()
            for i in range(len(self.warmup_local_scheduler_list)):
                self.warmup_local_scheduler_list[i].step()
        else:
            self.train_scheduler.step(epoch)
            for i in range(len(self.train_local_scheduler_list)):
                self.train_local_scheduler_list[i].step(epoch)

    def gan_scheduler_step(self, epoch = 0):
        for i in range(len(self.gan_scheduler_list)):
            self.gan_scheduler_list[i].step(epoch)
    
    def train_target_step(self, x_private, label_private, client_id=0):
        self.model.cloud.train()
        self.classifier.train()
        if "V" in self.scheme:
            self.model.local_list[client_id].train()
        else:
            self.f.train()
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Freeze batchnorm parameter of the client-side model.
        if self.load_from_checkpoint and self.finetune_freeze_bn:
            if client_id == 0:
                freeze_model_bn(self.f)
            elif client_id == 1:
                freeze_model_bn(self.c)
            else:
                freeze_model_bn(self.model.local_list[client_id])


        # Final Prediction Logits (complete forward pass)
        if client_id == 0:
            z_private = self.f(x_private)
        elif client_id == 1:
            z_private = self.c(x_private)
        else:
            z_private = self.model.local_list[client_id](x_private)

        if self.local_DP:
            if "laplace" in self.regularization_option:
                noise = torch.from_numpy(
                    np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=z_private.size())).cuda()
                z_private = z_private + noise.detach().float()
            else:  # apply gaussian noise
                delta = 10e-5
                sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                noise = sigma * torch.randn_like(z_private).cuda()
                z_private = z_private + noise.detach().float()
        if self.dropout_defense:
            z_private = dropout_defense(z_private, self.dropout_ratio)
        if self.topkprune:
            z_private = prune_defense(z_private, self.topkprune_ratio)
        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)
            if DENORMALIZE_OPTION:
                x_private = denormalize(x_private, self.dataset)
            
            if self.gan_loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif self.gan_loss_type == "MSE":
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)  
            z_private = z_private - grad.detach() * epsilon

        output = self.model.cloud(z_private)

        if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        
        if "label_smooth" in self.regularization_option:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.regularization_strength)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if client_id >= self.num_client_regularize:  # Do not apply regularization
            pass
        else:  # Do regularization
            if self.nopeek:
                #
                if "ttitcombe" in self.regularization_option:
                    dc = DistanceCorrelationLoss()
                    dist_corr_loss = self.alpha1 * dc(x_private, z_private)
                else:
                    dist_corr_loss = self.alpha1 * dist_corr(x_private, z_private).sum()

                total_loss = total_loss + dist_corr_loss
            if self.gan_regularizer and not self.gan_noise:
                self.local_AE_list[client_id].eval()
                output_image = self.local_AE_list[client_id](z_private)
                if DENORMALIZE_OPTION:
                    x_private = denormalize(x_private, self.dataset)
                '''MSE is poor in regularization here. unstable. We stick to SSIM'''
                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    ssim_term = ssim_loss(output_image, x_private)
                    
                    if self.ssim_threshold > 0.0:
                        if ssim_term > self.ssim_threshold:
                            gan_loss = self.alpha2 * (ssim_term - self.ssim_threshold) # Let SSIM approaches 0.4 to avoid overfitting
                        else:
                            gan_loss = 0.0 # Let SSIM approaches 0.4 to avoid overfitting
                    else:
                        gan_loss = self.alpha2 * ssim_term 
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    mse_term = mse_loss(output_image, x_private)
                    gan_loss = - self.alpha2 * mse_term  
                if "inv" in self.regularization_option:
                    gan_loss = - gan_loss
                total_loss = total_loss + gan_loss


        if "l1" in self.regularization_option:
            all_params = torch.cat([x.view(-1) for x in self.model.local_list[client_id].parameters()])
            l1_regularization = self.regularization_strength * torch.norm(all_params, 1)
            total_loss = total_loss + l1_regularization
        if "l2" in self.regularization_option:
            all_params = torch.cat([x.view(-1) for x in self.model.local_list[client_id].parameters()])
            l2_regularization = self.regularization_strength * torch.norm(all_params, 2)
            total_loss = total_loss + l2_regularization

        if "use_ce_loss" in self.regularization_option:
            pass
        
        if "gradient_noise_cloud" in self.regularization_option:
            for i, p in enumerate(self.model.cloud.parameters()):
                p.register_hook(lambda grad: torch.add(grad, self.regularization_strength * torch.rand_like(grad).cuda()))
        if "gradient_noise_local" in self.regularization_option:
            for i, p in enumerate(self.model.local_list[0].parameters()):
                p.register_hook(lambda grad: torch.add(grad, self.regularization_strength * torch.rand_like(grad).cuda()))

        total_loss.backward()


        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses
    



    def gradcollect_train_target_step(self, x_private, label_private, client_id=0):

        self.model.merge_classifier_cloud()
        self.model.cloud.train()
        self.classifier.train()
        self.model.local_list[client_id].train()
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Final Prediction Logits (complete forward pass)
        z_private = self.model.local_list[client_id](x_private)
        
        z_private.retain_grad()

        # hook all layer gradients
        # gradients_dict = {}
        # def get_gradients(name):
        #     def hook(model, input, output):
        #         gradients_dict[name + "-grad"] = output[0].detach().cpu().numpy()

        #     return hook
        # for name, m in self.model.cloud.named_modules():
        #     m.register_backward_hook(get_gradients("Gradients-Server-{}-{}".format(name, str(m).split("(")[0])))

        output = self.model.cloud(z_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        total_loss.backward()
        zeroing_grad(self.model.local_list[client_id])


        if not os.path.isdir(self.save_dir + "saved_grads"):
            os.makedirs(self.save_dir + "saved_grads")
        torch.save(z_private.grad.detach().cpu(), self.save_dir + f"saved_grads/grad_image{self.soft_image_id}_label{self.soft_train_count}.pt")
        # f = open(self.save_dir + f"saved_grads/grad_image{self.soft_image_id}_label{self.soft_train_count}.pkl","wb")
        # pickle.dump(gradients_dict, f)
        # f.close()
        if self.soft_train_count == 0:
            torch.save(x_private.detach().cpu(), self.save_dir + f"saved_grads/image_{self.soft_image_id}.pt")
            torch.save(label_private.detach().cpu(), self.save_dir + f"saved_grads/label_{self.soft_image_id}.pt")
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss
        self.model.unmerge_classifier_cloud()
        return total_losses, f_losses
    
    def craft_train_target_step(self, client_id):
        lambda_TV = 0.0
        lambda_l2 = 0.0
        craft_LR = 1e-1
        criterion = torch.nn.CrossEntropyLoss()

        image_save_path = self.save_dir + '/craft_pairs/'

        if not os.path.isdir(image_save_path):
            os.makedirs(image_save_path)

        if self.craft_step_count == 0 and self.craft_image_id == 0:
            image_shape = [self.batch_size, 3, 32, 32]
            label_shape = [self.batch_size, ]
            fake_image = torch.rand(image_shape, requires_grad=True, device="cuda")
            fake_label = torch.randint(low=0, high = self.num_class, size = label_shape, device="cuda")
        
        elif self.craft_step_count == self.num_craft_step:
            # save latest crafted image
            fake_image = torch.load(image_save_path + f'image_{self.craft_image_id}.pt')
            imgGen = fake_image.clone()
            if DENORMALIZE_OPTION:
                imgGen = denormalize(imgGen, self.dataset)
            torchvision.utils.save_image(imgGen, image_save_path + '/visual_{}.jpg'.format(self.craft_image_id))

            # reset counters
            self.craft_step_count = 0
            self.craft_image_id += 1
            image_shape = [self.batch_size, 3, 32, 32]
            label_shape = [self.batch_size, ]
            fake_image = torch.rand(image_shape, requires_grad=True, device="cuda")
            fake_label = torch.randint(low=0, high = self.num_class, size = label_shape, device="cuda")
        else:
            fake_image = torch.load(image_save_path + f'image_{self.craft_image_id}.pt').cuda()
            fake_label = torch.load(image_save_path + f'label_{self.craft_image_id}.pt').cuda()
        
        craft_optimizer = torch.optim.Adam(params=[fake_image], lr=craft_LR, amsgrad=True, eps=1e-3) # craft_LR = 1e-1 by default
                
        craft_optimizer.zero_grad()

        z_private = self.model.local_list[client_id](fake_image)  # Simulate Original
        z_private.retain_grad()
        output = self.model.cloud(z_private)

        if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        featureLoss = criterion(output, fake_label)

        TVLoss = TV(fake_image)
        normLoss = l2loss(fake_image)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

        totalLoss.backward()

        zeroing_grad(self.model.local_list[client_id])

        craft_optimizer.step()

        torch.save(fake_image.detach().cpu(), image_save_path + f'image_{self.craft_image_id}.pt')
        torch.save(fake_label.cpu(), image_save_path + f'label_{self.craft_image_id}.pt')
        torch.save(z_private.grad.detach().cpu(), image_save_path + f'grad_{self.craft_image_id}.pt')

        self.craft_step_count += 1
        return totalLoss.detach().cpu().numpy(), featureLoss.detach().cpu().numpy()

    def gan_train_target_step(self, client_id, num_query, epoch, batch):

        #if enable poison option
        poison_option = False

        self.model.cloud.train()
        self.classifier.train()
        self.model.local_list[client_id].train()
        self.generator.cuda()
        self.generator.train()

        train_output_path = self.save_dir + "generator_train"
        if os.path.isdir(train_output_path):
            rmtree(train_output_path)
        os.makedirs(train_output_path)


        #Sample Random Noise
        z = torch.randn((num_query, self.nz)).cuda()
        
        B = self.batch_size// 2

        labels_l = torch.randint(low=0, high=self.num_class, size = [B, ]).cuda()
        labels_r = copy.deepcopy(labels_l).cuda()
        label_private = torch.stack([labels_l, labels_r]).view(-1)
        
        #Get fake image from generator
        x_private = self.generator(z, label_private) # pre_x returns the output of G before applying the activation

        if poison_option and batch % 2 == 1: #poison step
            label_private = torch.randint(low=0, high=self.num_class, size = [self.batch_size, ]).cuda()
            x_private = x_private.detach()
        
        if epoch % 5 == 0 and batch == 0:
            imgGen = x_private.clone()
            if DENORMALIZE_OPTION:
                imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(epoch,
                                                                                            batch * self.batch_size + self.batch_size))
        
        # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
        g_noise_out_dist = torch.mean(torch.abs(x_private[:B, :] - x_private[B:, :]))
        g_noise_z_dist = torch.mean(torch.abs(z[:B, :] - z[B:, :]).view(B,-1),dim=1)
        g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * self.noise_w


        z_private = self.model.local_list[client_id](x_private)

        output = self.model.cloud(z_private)

        if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
            output = F.avg_pool2d(output, 4)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        elif self.arch == "resnet20" or self.arch == "resnet32":
            output = F.avg_pool2d(output, 8)
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        else:
            output = output.view(output.size(0), -1)
            output = self.classifier(output)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss - g_noise

        total_loss.backward()

        if "nopoison" in self.regularization_option:
            zeroing_grad(self.model.local_list[client_id])
        elif "poison" in self.regularization_option:
            pass
        else: # zeroing grad by default
            zeroing_grad(self.model.local_list[client_id])
        

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses

    def validate_target(self, client_id=0):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        if client_id == 0:
            self.f.eval()
        elif client_id == 1:
            self.c.eval()
        elif client_id > 1:
            self.model.local_list[client_id].eval()
        self.model.cloud.eval()
        self.classifier.eval()
        criterion = nn.CrossEntropyLoss()

        activation_0 = {}

        def get_activation_0(name):
            def hook(model, input, output):
                activation_0[name] = output.detach()

            return hook

        for name, m in self.model.local_list[client_id].named_modules():
            m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))

        for name, m in self.model.cloud.named_modules():
            m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))


        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            activation_0 = {}
            # compute output
            with torch.no_grad():

                output = self.model.local_list[client_id](input)

                if self.bhtsne:
                    self.save_activation_bhtsne(output, target, input.size(0), "train", f"client{client_id}")
                    exit()

                '''Optional, Test validation performance with local_DP/dropout (apply DP during query)'''
                if self.local_DP:
                    if "laplace" in self.regularization_option:
                        noise = torch.from_numpy(
                            np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=output.size())).cuda()
                    else:  # apply gaussian noise
                        delta = 10e-5
                        sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                        noise = sigma * torch.randn_like(output).cuda()
                    output += noise
                if self.dropout_defense:
                    output = dropout_defense(output, self.dropout_ratio)
                if self.topkprune:
                    output = prune_defense(output, self.topkprune_ratio)
            if self.gan_noise:
                epsilon = self.alpha2
                
                self.local_AE_list[client_id].eval()
                fake_act = output.clone()
                grad = torch.zeros_like(output).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                if DENORMALIZE_OPTION:
                    input = denormalize(input, self.dataset)

                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, input)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, input)
                    loss.backward()
                    grad += torch.sign(fake_act.grad) 

                output = output - grad.detach() * epsilon
            with torch.no_grad():
                output = self.model.cloud(output)

                if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                    output = F.avg_pool2d(output, 4)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    output = F.avg_pool2d(output, 8)
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                else:
                    output = output.view(output.size(0), -1)
                    output = self.classifier(output)
                loss = criterion(output, target)

            if i == 0:
                try:
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)

                    # setup tensorboard
                    if self.save_activation_tensor:
                        save_tensor_path = self.save_dir + "/saved_tensors"
                        if not os.path.isdir(save_tensor_path):
                            os.makedirs(save_tensor_path)
                    for key, value in activation_0.items():
                        if "client" in key:
                            self.writer.add_histogram("local_act/{}".format(key), value.clone().cpu().data.numpy(), i)
                            if self.save_activation_tensor:
                                np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                        if "server" in key:
                            self.writer.add_histogram("server_act/{}".format(key), value.clone().cpu().data.numpy(), i)
                            if self.save_activation_tensor:
                                np.save(save_tensor_path + "/{}_{}.npy".format(key, i), value.clone().cpu().data.numpy())
                    
                    for name, m in self.model.local_list[client_id].named_modules():
                        handle = m.register_forward_hook(get_activation_0("ACT-client-{}-{}".format(name, str(m).split("(")[0])))
                        handle.remove()
                    for name, m in self.model.cloud.named_modules():
                        handle = m.register_forward_hook(get_activation_0("ACT-server-{}-{}".format(name, str(m).split("(")[0])))
                        handle.remove()
                except:
                    print("something went wrong adding histogram, ignore it..")

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            # prec1 = accuracy(output.data, target, compress_V4shadowlabel=self.V4shadowlabel, num_client=self.num_client)[0] #If V4shadowlabel is activated, add one extra step to process output back to orig_class
            prec1 = accuracy(output.data, target)[
                0]  # If V4shadowlabel is activated, add one extra step to process output back to orig_class
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time

            # if i % 50 == 0:
        self.logger.debug('Test (client-{0}):\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            client_id, loss=losses,
            top1=top1))
        # for name, param in self.model.local_list[client_id].named_parameters():
        #     self.writer.add_histogram("local_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        # for name, param in self.model.cloud.named_parameters():
        #     self.writer.add_histogram("server_params/{}".format(name), param.clone().cpu().data.numpy(), 1)
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'
                          .format(top1=top1))

        return top1.avg, losses.avg

    def infer_path_list(self, path_to_infer):
        split_list = path_to_infer.split("checkpoint_f")
        first_part = split_list[0]
        second_part = split_list[1]
        model_path_list = []
        for i in range(self.num_client):
            if i == 0:
                model_path_list.append(path_to_infer)
            elif i == 1:
                model_path_list.append(first_part + "checkpoint_c" + second_part)
            else:
                model_path_list.append(first_part + "checkpoint_local{}".format(i) + second_part)

        return model_path_list

    def resume(self, model_path_f=None):
        if model_path_f is None:
            try:
                if "V" in self.scheme:
                    checkpoint = torch.load(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                    model_path_list = self.infer_path_list(self.save_dir + "checkpoint_f_{}.tar".format(self.n_epochs))
                else:
                    checkpoint = torch.load(self.save_dir + "checkpoint_{}.tar".format(self.n_epochs))
                    # model_path_list = self.infer_path_list(self.save_dir + "checkpoint_200.tar")
            except:
                print("No valid Checkpoint Found!")
                return
        else:
            if "V" in self.scheme:
                model_path_list = self.infer_path_list(model_path_f)

        if "V" in self.scheme:
            if self.num_client < 10:
                for i in range(self.num_client):
                    print("load client {}'s local".format(i))
                    checkpoint_i = torch.load(model_path_list[i])
                    self.model.local_list[i].cuda()
                    self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
            else:
                for i in range(10):
                    print("load client {}'s local".format(i))
                    checkpoint_i = torch.load(model_path_list[i])
                    self.model.local_list[i].cuda()
                    self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
                for i in range(10, self.num_client):
                    checkpoint_i = torch.load(model_path_list[9])
                    self.model.local_list[i].cuda()
                    self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)
        else:
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.f = self.model.local
            self.f.cuda()

        try:
            self.call_resume = True
            print("load cloud")
            checkpoint = torch.load(self.save_dir + "checkpoint_cloud_{}.tar".format(self.n_epochs))
            self.model.cloud.cuda()
            self.model.cloud.load_state_dict(checkpoint, strict = False)
            print("load classifier")
            checkpoint = torch.load(self.save_dir + "checkpoint_classifier_{}.tar".format(self.n_epochs))
            self.classifier.cuda()
            self.classifier.load_state_dict(checkpoint, strict = False)
        except:
            print("might be old style saving, load entire model")
            checkpoint = torch.load(model_path_f)
            self.model.cuda()
            self.model.load_state_dict(checkpoint, strict = False)
            self.call_resume = True
            self.f = self.model.local
            self.f.cuda()
            self.model.cloud.cuda()
            self.classifier = self.model.classifier
            self.classifier.cuda()

    def sync_client(self, idxs_users = None):
        
        if idxs_users is not None:
            active_local_list = [self.model.local_list[idx] for idx in idxs_users]
        else:
            active_local_list = self.model.local_list

        global_weights = average_weights(active_local_list)
        
        # get weight divergence, for sampled_users, other unsampled user divergence will be 0.
        if self.divergence_aware:
            weight_divergence = [0.0 for i in range(self.num_client)]
            for key in global_weights.keys():
                if "running" in key or "num_batches" in key:
                    continue
                if idxs_users is not None:
                    for i in idxs_users:
                        divergence = torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
                        weight_divergence[i] += divergence
                else:
                    for i in range(self.num_client):
                        divergence = torch.linalg.norm(torch.flatten(self.model.local_list[i].state_dict()[key] - global_weights[key]).float(), dim = -1, ord = 2)
                        weight_divergence[i] += divergence
                        # print("key: {}, divergence: {}".format(key, divergence))
            div = torch.FloatTensor(weight_divergence)
        
        for i in range(self.num_client):
            if not self.divergence_aware:
                self.model.local_list[i].load_state_dict(global_weights)
            elif self.divergence_aware:
                mu = self.div_lambda * div[i].item()
                mu = 1 if mu >= 1 else mu
                # print("mu of client {} is {}".format(i, mu))
                for key in global_weights.keys():
                    self.model.local_list[i].state_dict()[key] = mu * self.model.local_list[i].state_dict()[key] + (1 - mu) * global_weights[key]

        # return global_weights

    def sync_decoder(self, idxs_users = None):
        # update global weights
        if idxs_users is not None:
            active_local_list = [self.local_AE_list[idx] for idx in idxs_users]
            global_weights  = average_weights(active_local_list)
        else:
            global_weights  = average_weights(self.local_AE_list)

        # update global weights
        for i in range(self.num_client):
            self.local_AE_list[i].load_state_dict(global_weights)

    def gan_train_step(self, input_images, client_id, loss_type="SSIM"):
        device = next(self.model.local_list[client_id].parameters()).device

        input_images = input_images.to(device)

        self.model.local_list[client_id].eval()

        z_private = self.model.local_list[client_id](input_images)

        self.local_AE_list[client_id].train()

        x_private, z_private = Variable(input_images).to(device), Variable(z_private)

        if DENORMALIZE_OPTION:
            x_private = denormalize(x_private, self.dataset)

        if self.gan_noise:
            epsilon = self.alpha2
            
            self.local_AE_list[client_id].eval()
            fake_act = z_private.clone()
            grad = torch.zeros_like(z_private).cuda()
            fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
            x_recon = self.local_AE_list[client_id](fake_act)

            if loss_type == "SSIM":
                ssim_loss = pytorch_ssim.SSIM()
                loss = ssim_loss(x_recon, x_private)
                loss.backward()
                grad -= torch.sign(fake_act.grad)
            elif loss_type == "MSE":
                MSE_loss = torch.nn.MSELoss()
                loss = MSE_loss(x_recon, x_private)
                loss.backward()
                grad += torch.sign(fake_act.grad)
            else:
                raise ("No such loss_type for gan train step")
            
            z_private = z_private - grad.detach() * epsilon
            
            self.local_AE_list[client_id].train()

        output = self.local_AE_list[client_id](z_private.detach())

        if loss_type == "SSIM":
            ssim_loss = pytorch_ssim.SSIM()
            loss = -ssim_loss(output, x_private)
        elif loss_type == "MSE":
            MSE_loss = torch.nn.MSELoss()
            loss = MSE_loss(output, x_private)
        else:
            raise ("No such loss_type for gan train step")
        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].zero_grad()

        loss.backward()

        for i in range(len(self.gan_optimizer_list)):
            self.gan_optimizer_list[i].step()

        losses = loss.detach().cpu().numpy()
        del loss

        return losses

    def __call__(self, log_frequency=500, verbose=False, progress_bar=True):
        log_frequency = self.num_batches
        self.logger.debug("Model's smashed-data size is {}".format(str(self.model.get_smashed_data_size())))
        best_avg_accu = 0.0
        if not self.call_resume:
            LOG = np.zeros((self.n_epochs * self.num_batches, self.num_client))
            #load pre-train models
            if self.load_from_checkpoint:
                checkpoint_dir = "./pretrained_models/{}_cutlayer_{}_bottleneck_{}_dataset_{}/".format(self.arch, self.cutting_layer, self.bottleneck_option, self.source_task)
                try:
                    checkpoint_i = torch.load(checkpoint_dir + "checkpoint_f_best.tar")
                except:
                    print("No valid Checkpoint Found!")
                    return
                if "V" in self.scheme:
                    for i in range(self.num_client):
                        print("load client {}'s local".format(i))
                        self.model.local_list[i].cuda()
                        self.model.local_list[i].load_state_dict(checkpoint_i)
                else:
                    self.model.cuda()
                    self.model.local.load_state_dict(checkpoint_i)
                    self.f = self.model.local
                    self.f.cuda()
                
                load_classfier = False
                if self.load_from_checkpoint_server:
                    print("load cloud")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_cloud_best.tar")
                    self.model.cloud.cuda()
                    self.model.cloud.load_state_dict(checkpoint)
                if load_classfier:
                    print("load classifier")
                    checkpoint = torch.load(checkpoint_dir + "checkpoint_classifier_best.tar")
                    self.classifier.cuda()
                    self.classifier.load_state_dict(checkpoint)
            
            if self.gan_regularizer:
                if self.num_client <= 10:
                    self.pre_GAN_train(30, range(self.num_client))
                else:
                    self.pre_GAN_train(30, [0, 1])
                    self.sync_decoder([0, 1])

            self.logger.debug("Real Train Phase: done by all clients, for total {} epochs".format(self.n_epochs))

            if self.save_more_checkpoints:
                epoch_save_list = [1, 2 ,5 ,10 ,20 ,50 ,100]
            else:
                epoch_save_list = []
            # If optimize_computation, set GAN updating frequency to 1/5.
            ssim_log = 0.
            
            interval = self.optimize_computation
            self.logger.debug("GAN training interval N (once every N step) is set to {}!".format(interval))
            
            m = max(int(self.client_sample_ratio * self.num_client), 1)

            #Main Training
            self.logger.debug("Train in {} style".format(self.scheme))
            

            if "imagenet" not in self.dataset:
                dl_transforms = torch.nn.Sequential(
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                )
            else:
                dl_transforms = torch.nn.Sequential(
                    transforms.RandomCrop(224, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15)
                )

            if self.GM_train_option:
                if self.GM_data_proportion == 0.0:
                    print("TO use GM_train_option, Must have some data available")
                    exit()
                else:
                    if "CIFAR100" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_cifar100_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/self.GM_data_proportion))
                    elif "CIFAR10" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/self.GM_data_proportion))
                    elif "SVHN" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_SVHN_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/self.GM_data_proportion))
                    else: # default use cifar10
                        knockoff_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/self.GM_data_proportion))
                    knockoff_loader = knockoff_loader_list[0]
                    self.client_dataloader.append(knockoff_loader)

            # save iterator process of actual number of clients
            saved_iterator_list = []
            for client_id in range(self.actual_num_users):
                saved_iterator_list.append(iter(self.client_dataloader[client_id]))


            for epoch in range(1, self.n_epochs+1):
                if self.pre_train_ganadv and self.gan_regularizer:
                    if epoch > self.warm:
                        # self.scheduler_step(epoch + self.pretrain_epoch)
                        self.scheduler_step(epoch)
                        if self.gan_regularizer:
                            self.gan_scheduler_step(epoch + self.pretrain_epoch)
                else:
                    if epoch > self.warm:
                        self.scheduler_step(epoch)
                        if self.gan_regularizer:
                            self.gan_scheduler_step(epoch)
                
                if self.client_sample_ratio  == 1.0:
                    idxs_users = range(self.num_client)
                else:
                    idxs_users = np.random.choice(range(self.actual_num_users), self.num_client, replace=False) # 10 out of 1000
                
                # sample num_client for parapllel training from actual number of users, take their iterator as well.
                client_iterator_list = []
                for client_id in range(self.num_client):
                    if (self.Generator_train_option or self.Craft_train_option) and idxs_users[client_id] == self.actual_num_users - 1:
                        pass
                    else:
                        client_iterator_list.append(saved_iterator_list[idxs_users[client_id]])
            
                
                ## Secondary Loop
                for batch in range(self.num_batches):

                    # shuffle_client_list = range(self.num_client)
                    for client_id in range(self.num_client):
                        if self.scheme == "V1_epoch" or self.scheme == "V3_epoch":
                            self.optimizer_zero_grad()

                        # Get data
                        if (self.Generator_train_option or self.Craft_train_option) and idxs_users[client_id] == self.actual_num_users - 1:   # Data free no data.
                            
                            images, labels = None, None
                        
                        elif self.trigger_stop == False and self.Soft_train_option and epoch > self.gan_train_start_epoch and idxs_users[client_id] == self.actual_num_users - 1:  # SoftTrain, rotate labels
                            if self.soft_image_id == 0 and self.soft_train_count == 0:
                                client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                try:
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                except StopIteration:
                                    self.trigger_stop = True
                                    self.logger.debug("trigger_stop set to true")
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    old_images, old_labels = next(client_iterator_list[client_id])

                            elif self.soft_train_count == self.num_class:
                                self.soft_train_count = 0
                                try:
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                except StopIteration:
                                    self.trigger_stop = True
                                    self.logger.debug("trigger_stop set to true")
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                self.soft_image_id += 1
                            else:
                                old_labels = (old_labels + 1) % self.num_class # add 1 to label
                            
                            images, labels = old_images, old_labels

                        elif self.GM_train_option and idxs_users[client_id] == self.actual_num_users - 1:
                            if self.trigger_stop == False and epoch > self.gan_train_start_epoch:
                                
                                if self.soft_image_id == 0 and self.soft_train_count == 0:
                                    # produce new image 
                                    try:
                                        old_images, old_labels = next(client_iterator_list[client_id])
                                    except StopIteration:
                                        self.trigger_stop = True
                                        self.logger.debug("trigger_stop set to true")
                                        client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                        old_images, old_labels = next(client_iterator_list[client_id])
                                    
                                    old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)
                                elif self.soft_train_count == self.num_class:
                                    self.soft_train_count = 0
                                    try:
                                        old_images, old_labels = next(client_iterator_list[client_id])
                                    except StopIteration:
                                        self.trigger_stop = True
                                        self.logger.debug("trigger_stop set to true")
                                        client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                        old_images, old_labels = next(client_iterator_list[client_id])
                                    old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)
                                    self.soft_image_id += 1
                                else:
                                    old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)

                                images, labels = old_images, old_labels
                            
                            else:
                                images, labels = None, None
                        
                        elif self.trigger_stop == False and self.Normal_train_option and epoch > self.gan_train_start_epoch and idxs_users[client_id] == self.actual_num_users - 1:
                          
                            if self.soft_image_id == 0 and self.soft_train_count == 0:
                                client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                try:
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                except StopIteration:
                                    self.trigger_stop = True
                                    self.logger.debug("trigger_stop set to true")
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    old_images, old_labels = next(client_iterator_list[client_id])

                                old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)

                            elif self.soft_train_count == self.num_class:
                                self.soft_train_count = 0
                                try:
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                except StopIteration:
                                    self.trigger_stop = True
                                    self.logger.debug("trigger_stop set to true")
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    old_images, old_labels = next(client_iterator_list[client_id])
                                old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)
                                self.soft_image_id += 1
                            else:
                                old_labels = torch.ones_like(old_labels) * int(self.soft_train_count)
                        
                            images, labels = old_images, old_labels
                        
                        else: # Normal Case 
                            try:
                                images, labels = next(client_iterator_list[client_id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                    images, labels = next(client_iterator_list[client_id])
                            except StopIteration:
                                client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
                                images, labels = next(client_iterator_list[client_id])

                            if dl_transforms is not None:
                                images = dl_transforms(images)

                        

                        # Train the AE decoder if self.gan_regularizer is enabled:
                        if self.gan_regularizer and batch % interval == 0:
                            for i in range(self.gan_num_step):
                                ssim_log = -self.gan_train_step(images, client_id, loss_type=self.gan_loss_type)  # orig_epoch_gan_train

                        if self.scheme == "V2_epoch" or self.scheme == "V4_epoch" or self.scheme == "orig_epoch":
                            self.optimizer_zero_grad()
                        

                        if idxs_users[client_id] == self.actual_num_users - 1:
                            if epoch > self.gan_train_start_epoch:
                                
                                if self.Craft_train_option:
                                    train_loss, f_loss = self.craft_train_target_step(client_id)
                                elif self.Generator_train_option:
                                    train_loss, f_loss = self.gan_train_target_step(client_id, self.batch_size, epoch, batch)
                                elif self.GM_train_option:
                                    if self.trigger_stop == False:
                                        train_loss, f_loss = self.gradcollect_train_target_step(images, labels, client_id)
                                    else:
                                        pass
                                
                                elif self.trigger_stop == False and self.Normal_train_option:
                                    train_loss, f_loss = self.gradcollect_train_target_step(images, labels, client_id)
                                elif self.trigger_stop == False and self.Soft_train_option:
                                    train_loss, f_loss = self.gradcollect_train_target_step(images, labels, client_id)
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id)
                            else:
                                if self.Generator_train_option or self.GM_train_option or self.Craft_train_option :
                                    pass # do nothing is prior to the starting epoch   
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id)
                        else:
                            # Train step (client/server)
                            train_loss, f_loss = self.train_target_step(images, labels, client_id)
                        


                        if self.scheme == "V2_epoch" or self.scheme == "V4_epoch" or self.scheme == "orig_epoch":
                            self.optimizer_step()
                        
                        # Logging
                        # LOG[batch, client_id] = train_loss
                        if verbose and batch % log_frequency == 0:
                            self.logger.debug(
                                "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                    epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss, f_loss))
                            if self.gan_regularizer:
                                self.logger.debug(
                                    "log--[{}/{}][{}/{}][client-{}] Adversarial Loss of local AE: {:1.4f}".format(epoch,
                                                                                                            self.n_epochs,
                                                                                                            batch,
                                                                                                            self.num_batches,
                                                                                                            client_id,
                                                                                                            ssim_log))
                        if batch == 0:
                            self.writer.add_scalar('train_loss/client-{}/total'.format(client_id), train_loss,
                                                    epoch)
                            self.writer.add_scalar('train_loss/client-{}/cross_entropy'.format(client_id), f_loss,
                                                    epoch)
                        
                        if (self.Soft_train_option or self.GM_train_option or self.Normal_train_option) and epoch > self.gan_train_start_epoch and idxs_users[client_id] == self.actual_num_users - 1:  # SoftTrain, rotate labels
                            self.soft_train_count += 1
                # V1/V2 synchronization
                if self.scheme == "V1_epoch" or self.scheme == "V2_epoch":
                    self.sync_client()
                    if self.gan_regularizer and self.decoder_sync:
                        self.sync_decoder()
                    
                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)

                # Validate and get average accu among clients
                avg_accu = 0
                if "V1" in self.scheme or "V2" in self.scheme:
                    avg_accu, loss = self.validate_target(client_id=0)
                    self.writer.add_scalar('valid_loss/client-0/cross_entropy', loss, epoch)
                else:
                    for client_id in range(self.num_client):
                        accu, loss = self.validate_target(client_id=client_id)
                        self.writer.add_scalar('valid_loss/client-{}/cross_entropy'.format(client_id), loss, epoch)
                        avg_accu += accu
                    avg_accu = avg_accu / self.num_client

                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    best_avg_accu = avg_accu

                # Save Model regularly
                if epoch % 50 == 0 or epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)
                    if self.Generator_train_option:
                        torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))
        

        if self.Generator_train_option:
            self.generator.cuda()
            self.generator.eval()
            z = torch.randn((10, self.nz)).cuda()
            train_output_path = "{}/generator_final".format(self.save_dir)
            for i in range(self.num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                #Get fake image from generator
                fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

                imgGen = fake.clone()
                if DENORMALIZE_OPTION:
                    imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path):
                    os.mkdir(train_output_path)
                if not os.path.isdir(train_output_path + "/{}".format(self.n_epochs)):
                    os.mkdir(train_output_path + "/{}".format(self.n_epochs))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(self.n_epochs,"final_label{}".format(i)))
            torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))

        if not self.call_resume:
            self.logger.debug("Best Average Validation Accuracy is {}".format(best_avg_accu))
        else:
            LOG = None
            avg_accu = 0
            if "V1" in self.scheme or "V2" in self.scheme:
                avg_accu, _ = self.validate_target(client_id=0)
            else:
                for client_id in range(self.num_client):
                    accu, _ = self.validate_target(client_id=client_id)
                    avg_accu += accu
                avg_accu = avg_accu / self.num_client
            self.logger.debug("Best Average Validation Accuracy is {}".format(avg_accu))
        return LOG

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"

        if "V" in self.scheme:
            torch.save(self.f.state_dict(), self.save_dir + 'checkpoint_f_{}.tar'.format(epoch))
            if self.num_client > 1:
                torch.save(self.c.state_dict(), self.save_dir + 'checkpoint_c_{}.tar'.format(epoch))
            torch.save(self.model.cloud.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))
            if self.num_client > 2 and self.num_client < 10:
                for i in range(2, self.num_client):
                    torch.save(self.model.local_list[i].state_dict(),
                                self.save_dir + 'checkpoint_local{}_{}.tar'.format(i, epoch))
            elif self.num_client >= 10:
                for i in range(2, 10):
                    torch.save(self.model.local_list[i].state_dict(),
                                self.save_dir + 'checkpoint_local{}_{}.tar'.format(i, epoch))
        else:
            torch.save(self.model.state_dict(), self.save_dir + 'checkpoint_{}.tar'.format(epoch))
            torch.save(self.model.cloud.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))
            torch.save(self.classifier.state_dict(), self.save_dir + 'checkpoint_classifier_{}.tar'.format(epoch))

    def gen_ir(self, val_single_loader, local_model, img_folder="./tmp", intermed_reps_folder="./tmp", all_label=True,
               select_label=0, attack_from_later_layer=-1, attack_option = "MIA"):
        """
        Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
        """

        # switch to evaluate mode
        local_model.eval()
        file_id = 0
        for i, (input, target) in enumerate(val_single_loader):
            # input = input.cuda(async=True)
            input = input.cuda()
            target = target.item()
            if not all_label:
                if target != select_label:
                    continue

            img_folder = os.path.abspath(img_folder)
            intermed_reps_folder = os.path.abspath(intermed_reps_folder)
            if not os.path.isdir(intermed_reps_folder):
                os.makedirs(intermed_reps_folder)
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            # compute output
            with torch.no_grad():
                ir = local_model(input)
            
            if self.confidence_score:
                self.model.cloud.eval()
                ir = self.model.cloud(ir)
                if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                    ir = F.avg_pool2d(ir, 4)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    ir = F.avg_pool2d(ir, 8)
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
                else:
                    ir = ir.view(ir.size(0), -1)
                    ir = self.classifier(ir)
            
            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_4 = {}

                def get_activation_4(name):
                    def hook(model, input, output):
                        activation_4[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_4 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(ir)
                try:
                    ir = activation_4[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            ir = ir.float()

            if "truncate" in attack_option:
                try:
                    percentage_left = int(attack_option.split("truncate")[1])
                except:
                    print("auto extract percentage fail. Use default percentage_left = 20")
                    percentage_left = 20
                ir = prune_top_n_percent_left(ir, percentage_left)

            inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
            out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
            if DENORMALIZE_OPTION:
                input = denormalize(input, self.dataset)
            save_image(input, inp_img_path)
            torch.save(ir.cpu(), out_tensor_path)
            file_id += 1
        print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                   int(file_id * 0.1)))

    def pre_GAN_train(self, num_epochs, select_client_list=[0], server_option = False):
        # if enable server_option, pre-train the server_AE using public dataset at the server side., loging in client-0
        # Generate latest images/activation pair for all clients:
        if not server_option:
            client_iterator_list = []
            for client_id in range(self.num_client):
                client_iterator_list.append(iter(self.client_dataloader[client_id]))
            try:
                images, labels = next(client_iterator_list[client_id])
                if images.size(0) != self.batch_size:
                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                    images, labels = next(client_iterator_list[client_id])
            except StopIteration:
                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                images, labels = next(client_iterator_list[client_id])
        else:
            server_iterator = iter(self.pub_dataloader)
            try:
                images, labels = next(server_iterator)
                if images.size(0) != self.batch_size:
                    server_iterator = iter(self.pub_dataloader)
                    images, labels = next(server_iterator)
            except StopIteration:
                server_iterator = iter(self.pub_dataloader)
                images, labels = next(server_iterator)
            select_client_list = [0]
        
        for client_id in select_client_list:
            self.save_image_act_pair(images, labels, client_id, 0, clean_option=True)

        for client_id in select_client_list:

            attack_batchsize = 32
            attack_num_epochs = num_epochs
            model_log_file = self.save_dir + '/MIA_attack_{}_{}.log'.format(client_id, client_id)
            logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), client_id, client_id),
                                  model_log_file, level=logging.DEBUG)
            # pass
            image_data_dir = self.save_dir + "/img"
            tensor_data_dir = self.save_dir + "/img"

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)

            if self.dataset == "cifar100":
                val_single_loader, _, _ = get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "cifar10":
                val_single_loader, _, _ = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "svhn":
                val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "mnist":
                _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "fmnist":
                _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "facescrub":
                _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "tinyimagenet":
                _, val_single_loader = get_tinyimagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
            elif self.dataset == "imagenet":
                val_single_loader = get_imagenet_testloader(batch_size=1, num_workers=4, shuffle=False)
            attack_path = self.save_dir + '/MIA_attack_{}to{}'.format(client_id, client_id)
            if not os.path.isdir(attack_path):
                os.makedirs(attack_path)
                os.makedirs(attack_path + "/train")
                os.makedirs(attack_path + "/test")
                os.makedirs(attack_path + "/tensorboard")
                os.makedirs(attack_path + "/sourcecode")
            train_output_path = "{}/train".format(attack_path)
            test_output_path = "{}/test".format(attack_path)
            tensorboard_path = "{}/tensorboard/".format(attack_path)
            model_path = "{}/model.pt".format(attack_path)
            path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                         "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

            logger.debug("Generating IR ...... (may take a while)")
            if not server_option:
                self.gen_ir(val_single_loader, self.model.local_list[client_id], image_data_dir, tensor_data_dir)
                decoder = self.local_AE_list[client_id]
            else:
                self.gen_ir(val_single_loader, self.server_clientmodel_copy, image_data_dir, tensor_data_dir)
                decoder = self.server_AE
            
            

            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            # Construct a dataset for training the decoder
            trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

            # Do real test on target's client activation (and test with target's client ground-truth.)
            sp_testloader = apply_transform_test(1,
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0),
                                                 self.save_dir + "/save_activation_client_{}_epoch_{}".format(client_id,
                                                                                                             0))

            # Perform Input Extraction Attack
            self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                        attack_batchsize)
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class)

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)



    def steal_attack(self, num_query = 10, num_epoch = 200, attack_client=0, attack_style = "TrainME_option", data_proportion = 0.2, noniid_ratio = 1.0, train_clas_layer = -1, surrogate_arch = "same"):
        
        self.validate_target(attack_client)
        
        
        if self.bottleneck_option == "None":
            if train_clas_layer < 0:
                self.surrogate_model = architectures.create_surrogate_model(self.arch, self.cutting_layer, self.num_class, 0, "same")
            else:
                self.surrogate_model = architectures.create_surrogate_model(self.arch, self.cutting_layer, self.num_class, train_clas_layer, surrogate_arch)
        else:
            self.surrogate_model = copy.deepcopy(self.model)
            train_clas_layer += 1

        self.model.resplit(train_clas_layer)

        if surrogate_arch == "longer":
            train_clas_layer += 1
        
        if surrogate_arch == "shorter":
            train_clas_layer -= 1
            if train_clas_layer == -1:
                print("train class layer is too small for shorter architecture")
                exit()
        self.surrogate_model.resplit(train_clas_layer)
        

        # length_clas = self.surrogate_model.length_clas
        # length_tail = self.surrogate_model.length_tail
        # print("Tail model has {} cuttable & non-trivial layer".format(length_tail))
        # print("Classifier model has {} cuttable & non-trivial layer".format(length_clas))

        self.surrogate_model.local.apply(init_weights)
        self.surrogate_model.cloud.apply(init_weights)
        self.surrogate_model.classifier.apply(init_weights)


        # you can always lower the requirement to allow training of client-side surrogate model.
        # to compare with frozen client-side model
        # we don't recommend doing that unless you have lots of data.

        learning_rate = self.lr

        milestones = sorted([int(step * num_epoch) for step in [0.2, 0.5, 0.8]])
        optimizer_option = "SGD"

        gradient_matching = False
        Copycat_option = False
        Knockoff_option = False
        resume_option = False

        if "Copycat_option" in attack_style:
            Copycat_option = True # perform Copycat, use auxiliary dataset, inference is required
        if "Knockoff_option" in attack_style:
            Knockoff_option = True # perform knockoff set, use auxiliary dataset, inference is required
        if "gradient_matching" in attack_style or "grad" in attack_style:
            gradient_matching = True
        
        if "Craft_option" in attack_style: # perform Image Crafting ME attack, its Train-time variant is Craft_train_option
            Craft_option = True
            
            if "Craft_option_resume" in attack_style:
                resume_option = True
            else:
                resume_option = False
            craft_LR = 1e-1
            if "step" in attack_style:
                num_craft_step = int(attack_style.split("step")[1])
            else:
                num_craft_step = 20
            num_image_per_class = num_query // num_craft_step // self.num_class

            if num_image_per_class == 0:
                num_craft_step = num_query // self.num_class
                num_image_per_class = 1
                if num_craft_step == 0:
                    print("number of query is tooo small, not going to work.")
            image_shape = (1, 3, 32, 32)
            lambda_TV = 0.0
            lambda_l2 = 0.0
        else:
            Craft_option = False

        if "Generator_option" in attack_style: # perform GAN ME attack, its Train-time variant is Generator_train_option
            Generator_option = True
            gradient_matching = False
            nz = 512
            if "nz256" in attack_style:
                nz = 256
            elif "nz128" in attack_style:
                nz = 128
            self.noise_w = 50.0
            if "Generator_option_resume" in attack_style:
                resume_option = True
            else:
                resume_option = False
            if "Generator_option_pred" in attack_style:
                pred_option = True

                self.generator = architectures.GeneratorC(nz=nz, num_classes = self.num_class, ngf=128, nc=3, img_size=32)
                # self.generator = architectures.GeneratorA(nz=nz, nc=3, img_size=32)
            else:
                pred_option = False 

             
                self.generator = architectures.GeneratorC(nz=nz, num_classes = self.num_class, ngf=128, nc=3, img_size=32)
            # later call we call train_generator to train self.generator
        else:
            Generator_option = False
            self.generator = None

        if "GM_option" in attack_style:
            GM_option = True # use auxiliary dataset perform pure Gradient Matching
            gradient_matching = True
            if "GM_option_resume" in attack_style:
                resume_option = True
            else:
                resume_option = False
        else:
            GM_option = False

        if "TrainME_option" in attack_style:
            TrainME_option = True # perform direct trainning on the surrogate model

            if "TrainME_option_resume" in attack_style:
                resume_option = True
            else:
                resume_option = False
        else:
            TrainME_option = False

        if "SoftTrain_option" in attack_style: # enable introspective learning (KD using explanation)
            SoftTrain_option = True

            if "SoftTrain_option_resume" in attack_style:
                resume_option = True
            else:
                resume_option = False
            
            soft_alpha = 0.9

            retain_grad_tensor = "act"
            soft_lambda = 1 - soft_alpha # control the regularization strength
        else:
            SoftTrain_option = False

        

        surrogate_params = []

        if train_clas_layer != -1:  # This only hold for VGG architecture

            w_out =  self.model.local_list[attack_client].state_dict()
            self.surrogate_model.local.load_state_dict(w_out)
            self.surrogate_model.local.eval()
            self.surrogate_model.cloud.train()
            surrogate_params += list(self.surrogate_model.cloud.parameters())
            self.logger.debug(len(surrogate_params))
        else:
            self.surrogate_model.local.train()
            self.surrogate_model.cloud.train()
            surrogate_params += list(self.surrogate_model.local.parameters()) 
            surrogate_params += list(self.surrogate_model.cloud.parameters())

        if len(surrogate_params) == 0:
            self.logger.debug("surrogate parameter got nothing, add dummy param to prevent error")
            dummy_param = torch.nn.Parameter(torch.zero(1,1))
            surrogate_params = dummy_param
        else:
            self.logger.debug("surrogate parameter has {} trainable parameters!".format(len(surrogate_params)))
            
        if optimizer_option == "SGD":
            self.suro_optimizer = torch.optim.SGD(surrogate_params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        else:
            self.suro_optimizer = torch.optim.Adam(surrogate_params, lr=0.0001, weight_decay=5e-4)
        

        if "Generator_option_pred" in attack_style:
            milestones = sorted([int(step * (num_query // 200)) for step in [0.2, 0.5, 0.8]])

        self.suro_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.suro_optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay

        # query getting data
        if data_proportion == 0.0:
            attacker_loader_list = [None]
        else:
            if self.dataset == "cifar100":
                attacker_loader_list, _, _ = get_cifar100_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion), noniid_ratio = noniid_ratio)
            elif self.dataset == "cifar10":
                attacker_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion), noniid_ratio = noniid_ratio)
            elif self.dataset == "imagenet":
                attacker_loader_list = get_imagenet_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion), noniid_ratio = noniid_ratio)
            elif self.dataset == "svhn":
                attacker_loader_list, _, _ = get_SVHN_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
            elif self.dataset == "mnist":
                attacker_loader_list, _= get_mnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
            elif self.dataset == "fmnist":
                attacker_loader_list, _= get_fmnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
            else:
                raise("Unknown Dataset!")

        attacker_dataloader = attacker_loader_list[attack_client]

        # prepare model, please make sure model.cloud and classifier are loaded from checkpoint.
        self.model.local_list[attack_client].cuda()
        self.model.local_list[attack_client].eval()
        self.model.cloud.cuda()
        self.model.cloud.eval()
        
        self.surrogate_model.local.cuda()
        self.surrogate_model.cloud.cuda()

        # prepare activation/grad/label pairs for the grad inversion
        criterion = torch.nn.CrossEntropyLoss()
        save_images = []
        save_grad = []
        save_label = []

        ''' crafting training dataset for surrogate model training'''

        if Craft_option:

            if resume_option:
                saved_crafted_image_path = self.save_dir + "craft_pairs/"
                if os.path.isdir(saved_crafted_image_path):
                    print("load from crafted during training")
                else:
                    print("No saved pairs is available!")
                    exit()
                for file in glob.glob(saved_crafted_image_path + "*"):
                    if "image" in file:
                        fake_image = torch.load(file)
                        image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                        fake_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")
                        fake_grad = torch.load(saved_crafted_image_path + f"grad_{image_id}.pt")

                        save_images.append(fake_image.clone())
                        save_grad.append(fake_grad.clone())
                        save_label.append(fake_label.clone())
            else:
                for c in range(self.num_class):
                    fake_label = c * torch.ones((1,)).long().cuda().view(1,)
                    
                    for i in range(num_image_per_class):
                        fake_image = torch.rand(image_shape, requires_grad=True, device="cuda")
                        # fake_image = torch.randn(image_shape, requires_grad=True).clamp_(-1, 1).cuda()
                        craft_optimizer = torch.optim.Adam(params=[fake_image], lr=craft_LR, amsgrad=True, eps=1e-3) # craft_LR = 1e-1 by default
                        for step in range(1, num_craft_step + 1):
                            craft_optimizer.zero_grad()

                            z_private = self.model.local_list[attack_client](fake_image)  # Simulate Original
                            z_private.retain_grad()
                            output = self.model.cloud(z_private)

                            featureLoss = criterion(output, fake_label)

                            TVLoss = TV(fake_image)
                            normLoss = l2loss(fake_image)

                            totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

                            totalLoss.backward()

                            craft_optimizer.step()
                            if step == 0 or step == num_craft_step:
                                self.logger.debug("Image {} class {} - Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(i, c, step,
                                                                                                    featureLoss.cpu().detach().numpy(),
                                                                                                    TVLoss.cpu().detach().numpy(),
                                                                                                    normLoss.cpu().detach().numpy()))
                        
                        save_images.append(fake_image.detach().cpu().clone())
                        save_grad.append(z_private.grad.detach().cpu().clone())
                        save_label.append(fake_label.cpu().clone())

        ''' TrainME: Use available training dataset for surrogate model training'''
        ''' Because of data augmentation, set query to 10 there'''

        if TrainME_option:
            
            
            if resume_option and TrainME_option and gradient_matching: #TODO: Test correctness.cluster all label together.
                assist_images = []
                assist_grad = []
                assist_label = []
                saved_crafted_image_path = self.save_dir + "saved_grads/"
                if os.path.isdir(saved_crafted_image_path):
                    print("load from data collected during training")
                else:
                    print("No saved data is presented!")
                    exit()
                
                max_image_id = 0
                for file in glob.glob(saved_crafted_image_path + "*"):
                    if "image" in file and "grad_image" not in file:
                        image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                        if image_id > max_image_id:
                            max_image_id = image_id
                for label in range(self.num_class): # put same label together
                    for file in glob.glob(saved_crafted_image_path + "*"):
                        if "image" in file and "grad_image" not in file:
                            image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                            if image_id >= max_image_id - 5:
                                saved_image = torch.load(file)
                                saved_grad = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{label}.pt")
                                saved_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")

                                assist_images.append(saved_image.clone())
                                assist_grad.append(saved_grad.clone()) # add a random existing grad.
                                assist_label.append(saved_label.clone())
            
            for images, labels in attacker_dataloader:
                images = images.cuda()
                labels = labels.cuda()

                self.optimizer_zero_grad()
                z_private = self.model.local_list[attack_client](images)
                z_private.retain_grad()

                if self.bhtsne:
                    valid_key = "z_private"
                    self.save_activation_bhtsne(z_private, labels, images.size(0), attack_style, valid_key)

                output = self.model.cloud(z_private)

                loss = criterion(output, labels)
                loss.backward(retain_graph = True)
                z_private_grad = z_private.grad.detach().cpu()
                
                save_images.append(images.cpu().clone())
                save_grad.append(z_private_grad.clone())
                save_label.append(labels.cpu().clone())
        
        ''' SoftTrainME, crafts soft input label pairs for surrogate model training'''
        if  SoftTrain_option:
            # Use SoftTrain_option, query the gradients on inputs with all label combinations
            similar_func = torch.nn.CosineSimilarity(dim = 1)
            if resume_option:
                saved_crafted_image_path = self.save_dir + "saved_grads/"
                if os.path.isdir(saved_crafted_image_path):
                    print("load from data collected during training")
                else:
                    print("No saved data is presented!")
                    exit()
                for file in glob.glob(saved_crafted_image_path + "*"):
                    if "image" in file and "grad_image" not in file:
                        
                        saved_image = torch.load(file)
        
                        image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                        true_grad = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label0.pt").cuda()
                        true_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt").cuda()
                        cos_sim_list = []

                        for c in range(self.num_class):
                            fake_grad = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{c}.pt").cuda()
                            cos_sim_val = similar_func(fake_grad.view(true_label.size(0), -1), true_grad.view(true_label.size(0), -1))
                            cos_sim_list.append(cos_sim_val.detach().clone()) # 10 item of [128, 1]

                        cos_sim_tensor = torch.stack(cos_sim_list).view(self.num_class, -1).t().cuda() # [128, 10]
                        cos_sim_tensor += 1
                        cos_sim_sum = (cos_sim_tensor).sum(1) - 1
                        derived_label = (1 - soft_alpha) * cos_sim_tensor / cos_sim_sum.view(-1, 1) # [128, 10]

                        
                        labels_as_idx = true_label.detach().view(-1, 1)
                        replace_val = soft_alpha * torch.ones(labels_as_idx.size(), dtype=torch.long).cuda()
                        derived_label.scatter_(1, labels_as_idx, replace_val)

                        save_images.append(saved_image.clone())
                        save_grad.append(true_grad.detach().cpu().clone())
                        save_label.append(derived_label.detach().cpu().clone())
            else:
                if num_query < self.num_class * 100:
                    print("Query budget is too low to run SoftTrainME")
                
                for i, (images, labels) in enumerate(attacker_dataloader):
                    self.optimizer_zero_grad()
                    images = images.cuda()
                    if retain_grad_tensor == "img":
                        images.requires_grad = True
                        images.retain_grad()
                    labels = labels.cuda()
                    cos_sim_list = []

                    
                    z_private = self.model.local_list[attack_client](images)
                    
                    
                    if retain_grad_tensor == "act":
                        z_private.retain_grad()
                        
                    output = self.model.cloud(z_private)

                    loss = criterion(output, labels)

                    # one_hot_target = F.one_hot(labels, num_classes=self.num_class)
                    # log_prob = torch.nn.functional.log_softmax(output, dim=1)
                    # loss = torch.mean(torch.sum(-one_hot_target * log_prob, dim=1))
                    loss.backward(retain_graph = True)

                    if retain_grad_tensor == "img":
                        z_private_grad = images.grad.detach().clone()
                    elif retain_grad_tensor == "act":
                        z_private_grad = z_private.grad.detach().clone()

                    if i * self.num_class * 100 <= num_query: # in query budget, craft soft label 
                        for c in range(self.num_class):
                            fake_label = c * torch.ones_like(labels).cuda()
                            self.optimizer_zero_grad()
                            if retain_grad_tensor == "act":
                                z_private.grad.zero_()
                            elif retain_grad_tensor == "img":
                                images.grad.zero_()
                            z_private = self.model.local_list[attack_client](images)

                            if retain_grad_tensor == "act":
                                z_private.retain_grad()
                            elif retain_grad_tensor == "img":
                                images.retain_grad()
                            output = self.model.cloud(z_private)

                            loss = criterion(output, fake_label)
                            
                            loss.backward(retain_graph = True)
                            if retain_grad_tensor == "act":
                                fake_z_private_grad = z_private.grad.detach().clone()
                            elif retain_grad_tensor == "img":
                                fake_z_private_grad = images.grad.detach().clone()
                            cos_sim_val = similar_func(fake_z_private_grad.view(labels.size(0), -1), z_private_grad.view(labels.size(0), -1))
                            cos_sim_list.append(cos_sim_val.detach().clone()) # 10 item of [128, 1]
                        cos_sim_tensor = torch.stack(cos_sim_list).view(self.num_class, -1).t().cuda()
                        cos_sim_tensor += 1
                        cos_sim_sum = (cos_sim_tensor).sum(1) - 1
                        derived_label = (1 - soft_alpha) * cos_sim_tensor / cos_sim_sum.view(-1, 1) # [128, 10]

                        
                        labels_as_idx = labels.detach().view(-1, 1)
                        replace_val = soft_alpha * torch.ones(labels_as_idx.size(), dtype=torch.long).cuda()
                        derived_label.scatter_(1, labels_as_idx, replace_val)

                    else:  # out of query budget, use hard label
                        derived_label = F.one_hot(labels, num_classes=self.num_class)

                    if retain_grad_tensor == "img":
                        images.requires_grad = False
                    save_images.append(images.cpu().clone())
                    save_grad.append(z_private_grad.cpu().clone())
                    save_label.append(derived_label.cpu().clone())
                    
        ''' Knockoffset, option_B has no prediction query (use grad-matching), option_C has predicion query (craft input-label pair)'''
        if GM_option or Copycat_option or Knockoff_option:
            # We fix query batch size to 100 to better control the total query number.

            if data_proportion == 0.0:
                knockoff_loader_list = [None]
            else:
                if "CIFAR100" in attack_style:
                    knockoff_loader_list, _, _ = get_cifar100_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
                elif "CIFAR10" in attack_style:
                    knockoff_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
                elif "SVHN" in attack_style:
                    knockoff_loader_list, _, _ = get_SVHN_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
                elif "MNIST" in attack_style:
                    knockoff_loader_list, _= get_mnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
                
                else: # default use cifar10
                    knockoff_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
            knockoff_loader = knockoff_loader_list[0]
        
        if GM_option and knockoff_loader is not None:

            if resume_option: #TODO: Test correctness. cluster all label together.
                saved_crafted_image_path = self.save_dir + "saved_grads/"
                if os.path.isdir(saved_crafted_image_path):
                    print("load from data collected during training")
                else:
                    print("No saved data is presented!")
                    exit()
                
                max_image_id = 0
                for file in glob.glob(saved_crafted_image_path + "*"):
                    if "image" in file and "grad_image" not in file:
                        image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                        if image_id > max_image_id:
                            max_image_id = image_id
                for label in range(self.num_class): # put same label together
                    for file in glob.glob(saved_crafted_image_path + "*"):
                        if "image" in file and "grad_image" not in file:
                            
            
                            image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))

                            if image_id > max_image_id - 1: # collect only the last two valid data batch. (even this is very bad)
                                saved_image = torch.load(file)
                                saved_grads = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{label}.pt")
                                saved_label = label * torch.ones(saved_grads.size(0), ).long()
                                # saved_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")

                                save_images.append(saved_image.clone())
                                save_grad.append(saved_grads.clone()) # add a random existing grad.
                                save_label.append(saved_label.clone())
            else:
                self.model.local_list[attack_client].eval()
                self.model.cloud.eval()
                for i, (inputs, target) in enumerate(knockoff_loader):
                    if i * self.num_class * 100 >= num_query: # limit grad query budget
                        break
                    inputs = inputs.cuda()
                    for j in range(self.num_class):
                        label = j * torch.ones_like(target).cuda()
                        self.optimizer_zero_grad()
                        
                        z_private = self.model.local_list[attack_client](inputs)
                        z_private.grad = None
                        z_private.retain_grad()
                        output = self.model.cloud(z_private)
                        
                        loss = criterion(output, label)
                        loss.backward(retain_graph = True)
                        
                        z_private_grad = z_private.grad.detach().cpu()
                        
                        save_grad.append(z_private_grad.clone()) # add a random existing grad.
                        
                        save_images.append(inputs.cpu().clone())
                        
                        save_label.append(label.cpu().clone())


        if Copycat_option and knockoff_loader is not None:
            # implement transfer set as copycat paper, here we use cifar-10 dataset.
            self.model.local_list[attack_client].eval()
            self.model.cloud.eval()
            self.classifier.eval()
            for i, (inputs, target) in enumerate(knockoff_loader):
                if i * 100 >= num_query:  # limit pred query budget
                    break
                inputs = inputs.cuda()
                with torch.no_grad():
                    z_private = self.model.local_list[attack_client](inputs)
                    output = self.model.cloud(z_private)
                    _, pred = output.topk(1, 1, True, True)

                    save_images.append(inputs.cpu().clone())
                    save_grad.append(torch.zeros((inputs.size(0), z_private.size(1), z_private.size(2), z_private.size(3)))) # add a random existing grad.
                    save_label.append(pred.view(-1).cpu().clone())
        
        if Knockoff_option and knockoff_loader is not None:
            # implement transfer set as knockoff paper, where attacker get access to confidence score
            self.model.local_list[attack_client].eval()
            self.model.cloud.eval()
            self.classifier.eval()
            for i, (inputs, target) in enumerate(knockoff_loader):
                if i * 100 >= num_query:   # limit pred query budget
                    break
                inputs = inputs.cuda()
                with torch.no_grad():
                    z_private = self.model.local_list[attack_client](inputs)
                    output = self.model.cloud(z_private)
                    softmax_layer = torch.nn.Softmax(dim=None)
                    log_prob = softmax_layer(output)
                    save_images.append(inputs.cpu().clone())
                    save_grad.append(torch.zeros((inputs.size(0), z_private.size(1), z_private.size(2), z_private.size(3)))) # add a random existing grad.
                    save_label.append(log_prob.cpu().clone())



        '''GAN_ME, data-free model extraction, train a conditional GAN, train-time option: use 'gan_train' in regularization_option'''
        if Generator_option:
            # get prototypical data using GAN, training generator consumes grad query.
            self.train_generator(num_query = num_query, nz = nz, 
                                    data_helper = attacker_dataloader, resume = resume_option,
                                    discriminator_option=False, pred_option = pred_option)
            if pred_option:
                exit()

        # Packing list to dataloader.
        if not Generator_option:

            if GM_option and resume_option:
                if_shuffle = False
                batch_size = 100
            else:
                if_shuffle = True
                batch_size = self.batch_size
            
            save_images = torch.cat(save_images)
            save_grad = torch.cat(save_grad)
            save_label = torch.cat(save_label)
            indices = torch.arange(save_label.shape[0]).long()
            ds = torch.utils.data.TensorDataset(indices, save_images, save_grad, save_label)

            

            dl = torch.utils.data.DataLoader(
                ds, batch_size=batch_size, num_workers=4, shuffle=if_shuffle
            )
            # print("length of dl is ", len(dl))
            mean_norm = save_grad.norm(dim=-1).mean().detach().item()
        else:
            self.generator.eval()
            self.generator.cuda()
            mean_norm = 0.0
            test_output_path = self.save_dir + "generator_test"
            if os.path.isdir(test_output_path):
                rmtree(test_output_path)
            os.makedirs(test_output_path)
        
        if resume_option and TrainME_option and gradient_matching: #TODO: Test correctness.
            assist_images = torch.cat(assist_images)
            assist_grad = torch.cat(assist_grad)
            assist_label = torch.cat(assist_label)
            indices = torch.arange(assist_label.shape[0]).long()
            ds_assist = torch.utils.data.TensorDataset(indices, assist_images, assist_grad, assist_label)

            dl_assist = torch.utils.data.DataLoader(
                ds_assist, batch_size=self.batch_size, num_workers=4, shuffle=True
            )
            print("length of dl_assist is ", len(dl_assist))
            mean_norm = assist_grad.norm(dim=-1).mean().detach().item()

        dl_transforms = torch.nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        )
        if Craft_option or Generator_option or GM_option:
            dl_transforms = None
        
        if SoftTrain_option and resume_option: #TODO: test the usefulness of this.
            dl_transforms = None
        

        min_grad_loss = 9.9
        acc_loss_min_grad_loss = 9.9
        val_acc_max = 0.0
        fidelity_max = 0.0
        
        best_tail_state_dict = None
        best_classifier_state_dict = None
        val_accu, fidel_score = self.steal_test(attack_client=attack_client)
        self.logger.debug("epoch: {}, val_acc: {}, val_fidelity: {}".format(0, val_accu, fidel_score))



        # Train surrogate model
        for epoch in range(1, num_epoch + 1):
            grad_loss_list = []
            acc_loss_list = []
            acc_list = []
            self.suro_scheduler.step(epoch)

            # Use grads only for training surrogate
            if GM_option: 
                acc_loss_list.append(0.0)
                acc_list.append(0.0)
                for idx, (index, image, grad, label) in enumerate(dl):
                    # if dl_transforms is not None:
                    #     image = dl_transforms(image)
                    image = image.cuda()
                    grad = grad.cuda()
                    label = label.cuda()
                    self.suro_optimizer.zero_grad()
                    act = self.surrogate_model.local(image)
                    act.requires_grad_()

                    output = self.surrogate_model.cloud(act)

                    ce_loss = criterion(output, label)

                    gradient_loss_style = "l2"

                    grad_lambda = 1.0

                    grad_approx = torch.autograd.grad(ce_loss, act, create_graph = True)[0]

                    #TODO: just test difference, delete below
                    # grad_approx = torch.autograd.grad(ce_loss, tail_output, create_graph = True)[0]

                    # grad_loss = ((grad - grad_approx).norm(dim=1, p =1)).mean() / mean_norm + torch.mean(1 - F.cosine_similarity(grad_approx, grad, dim=1))
                    if gradient_loss_style == "l2":
                        grad_loss = ((grad - grad_approx).norm(dim=1, p =2)).mean() / mean_norm
                    elif gradient_loss_style == "l1":
                        grad_loss = ((grad - grad_approx).norm(dim=1, p =1)).mean() / mean_norm
                    elif gradient_loss_style == "cosine":
                        grad_loss = torch.mean(1 - F.cosine_similarity(grad_approx, grad, dim=1))

                    total_loss = grad_loss * grad_lambda

                    
                    total_loss.backward()
                    self.suro_optimizer.step()

                    grad_loss_list.append(total_loss.detach().cpu().item())
                
            if not (Generator_option or GM_option): # dl contains input and label, very general framework
                for idx, (index, image, grad, label) in enumerate(dl):
                    if dl_transforms is not None:
                        image = dl_transforms(image)
                    
                    image = image.cuda()
                    grad = grad.cuda()
                    label = label.cuda()

                    self.suro_optimizer.zero_grad()
                    act = self.surrogate_model.local(image)
                    output = self.surrogate_model.cloud(act)

                    if Knockoff_option:
                        log_prob = torch.nn.functional.log_softmax(output, dim=1)
                        ce_loss = torch.mean(torch.sum(-label * log_prob, dim=1))
                    elif SoftTrain_option:
                        _, real_label = label.max(dim = 1)
                        ce_loss = (1 - soft_lambda) * criterion(output, real_label) + soft_lambda * torch.mean(torch.sum(-label * torch.nn.functional.log_softmax(output, dim=1), dim=1))
                    else:
                        ce_loss = criterion(output, label)

                    total_loss = ce_loss
                    total_loss.backward()
                    self.suro_optimizer.step()

                    acc_loss_list.append(ce_loss.detach().cpu().item())

                    if SoftTrain_option or Knockoff_option:
                        _, real_label = label.max(dim=1)
                        acc = accuracy(output.data, real_label)[0]
                    else:
                        acc = accuracy(output.data, label)[0]
                        
                    acc_list.append(acc.cpu().item())
                
                if resume_option and TrainME_option and gradient_matching: #TODO: use dl_asssit (gradient matching) to assit the training
                    for idx, (index, image, grad, label) in enumerate(dl_assist):
                        
                        image = image.cuda()
                        grad = grad.cuda()
                        label = label.cuda()

                        self.suro_optimizer.zero_grad()
                        act = self.surrogate_model.local(image)
                        output = self.surrogate_model.cloud(act)

                        ce_loss = criterion(output, label)

                        # grad_lambda controls the strength of gradient matching lass
                        # ce_loss_lower_bound controls when enters the gradient matching phase.
                        gradient_loss_style = "l2"
                        grad_lambda = 1.0

                        grad_approx = torch.autograd.grad(ce_loss, act, create_graph = True)[0]

                        # grad_loss = ((grad - grad_approx).norm(dim=1, p =1)).mean() / mean_norm + torch.mean(1 - F.cosine_similarity(grad_approx, grad, dim=1))
                        if gradient_loss_style == "l2":
                            grad_loss = ((grad - grad_approx).norm(dim=1, p =2)).mean() / mean_norm
                        elif gradient_loss_style == "l1":
                            grad_loss = ((grad - grad_approx).norm(dim=1, p =1)).mean() / mean_norm
                        elif gradient_loss_style == "cosine":
                            grad_loss = torch.mean(1 - F.cosine_similarity(grad_approx, grad, dim=1))


                        total_loss = grad_loss * grad_lambda
                        
                        total_loss.backward()
                        self.suro_optimizer.step()

                        grad_loss_list.append(grad_loss.detach().cpu().item())


            if Generator_option: # dynamically generates input and label using the trained Generator, used only in GAN-ME
                iter_generator_times = 20
                for i in range(iter_generator_times):
                    z = torch.randn((self.batch_size, nz)).cuda()
                    label = torch.randint(low=0, high=self.num_class, size = [self.batch_size, ]).cuda()
                    fake_input = self.generator(z, label)

                    #Save images to file
                    if epoch == 1:
                        imgGen = fake_input.clone()
                        if DENORMALIZE_OPTION:
                            imgGen = denormalize(imgGen, self.dataset)
                        if not os.path.isdir(test_output_path + "/{}".format(epoch)):
                            os.mkdir(test_output_path + "/{}".format(epoch))
                        torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(epoch,
                                                                                                        i * self.batch_size + self.batch_size))
                    self.suro_optimizer.zero_grad()

                    output = self.surrogate_model.local(fake_input)
                    output = self.surrogate_model.cloud(output)

                    ce_loss = criterion(output, label)

                    total_loss = ce_loss
                    
                    total_loss.backward()

                    self.suro_optimizer.step()
                    
                    acc_loss_list.append(ce_loss.detach().cpu().item())
                    acc = accuracy(output.data, label)[0]
                    acc_list.append(acc.cpu().item())
            

            if gradient_matching:
                grad_loss_mean = np.mean(grad_loss_list)
            else:
                grad_loss_mean = None
                min_grad_loss = None
            acc_loss_mean = np.mean(acc_loss_list)
            avg_acc = np.mean(acc_list)

            val_accu, fidel_score = self.steal_test(attack_client=attack_client)

            if gradient_matching:
                if val_accu > val_acc_max:
                    min_grad_loss = grad_loss_mean
                    acc_loss_min_grad_loss = acc_loss_mean
                    val_acc_max = val_accu
                    acc_max_fidelity = fidel_score
                    best_client_state_dict = self.surrogate_model.local.state_dict()
                    best_tail_state_dict = self.surrogate_model.cloud.state_dict()
                if fidel_score > fidelity_max:
                    min_grad_loss = grad_loss_mean
                    acc_loss_min_grad_loss = acc_loss_mean
                    fidelity_max = fidel_score
                    fidel_max_acc = val_accu
                    closest_client_state_dict = self.surrogate_model.local.state_dict()
                    closest_tail_state_dict = self.surrogate_model.cloud.state_dict()
            else:
                if val_accu > val_acc_max:
                    acc_loss_min_grad_loss = acc_loss_mean
                    val_acc_max = val_accu
                    acc_max_fidelity = fidel_score
                    best_client_state_dict = self.surrogate_model.local.state_dict()
                    best_tail_state_dict = self.surrogate_model.cloud.state_dict()
                if fidel_score > fidelity_max:
                    acc_loss_min_grad_loss = acc_loss_mean
                    fidelity_max = fidel_score
                    fidel_max_acc = val_accu
                    closest_client_state_dict = self.surrogate_model.local.state_dict()
                    closest_tail_state_dict = self.surrogate_model.cloud.state_dict()
            
            self.logger.debug("epoch: {}, train_acc: {}, val_acc: {}, fidel_score: {}, acc_loss: {}, grad_loss: {}".format(epoch, avg_acc, val_accu, fidel_score, acc_loss_mean, grad_loss_mean))
        

        

        if gradient_matching:
            if best_tail_state_dict is not None:
                self.logger.debug("load best stealed model.")
                self.logger.debug("Best perform model, val_acc: {}, fidel_score: {}, acc_loss: {}, grad_loss: {}".format(val_acc_max, acc_max_fidelity, acc_loss_min_grad_loss, min_grad_loss))
                self.surrogate_model.local.load_state_dict(best_client_state_dict)
                self.surrogate_model.cloud.load_state_dict(best_tail_state_dict)
            
            if closest_tail_state_dict is not None:
                self.logger.debug("load cloest stealed model.")
                self.logger.debug("Cloest perform model, val_acc: {}, fidel_score: {}, acc_loss: {}, grad_loss: {}".format(fidel_max_acc, fidelity_max, acc_loss_min_grad_loss, min_grad_loss))
                self.surrogate_model.local.load_state_dict(closest_client_state_dict)
                self.surrogate_model.cloud.load_state_dict(closest_tail_state_dict)
        
        else:
            self.logger.debug("load best stealed model.")
            self.logger.debug("Best perform model, val_acc: {}, fidel_score: {}, acc_loss: {}".format(val_acc_max, acc_max_fidelity, acc_loss_min_grad_loss))
            self.surrogate_model.local.load_state_dict(best_client_state_dict)
            self.surrogate_model.cloud.load_state_dict(best_tail_state_dict)

            self.logger.debug("load cloest stealed model.")
            self.logger.debug("Closest perform model, val_acc: {}, fidel_score: {}, acc_loss: {}".format(fidel_max_acc, fidelity_max, acc_loss_min_grad_loss))
            self.surrogate_model.local.load_state_dict(closest_client_state_dict)
            self.surrogate_model.cloud.load_state_dict(closest_tail_state_dict)
 
        #TODO: perform adversarial attack using surrogate model



    def train_generator(self, num_query, nz, data_helper = None, resume = False, discriminator_option = False, pred_option = False):
        
        lr_G = 1e-4
        lr_C = 1e-4
        d_iter = 5
        
        if pred_option:
            num_steps = num_query // 100 // (1 + d_iter) # train g_iter + d_iter times
        else:
            num_steps = num_query // 100 # train once
        steps = sorted([int(step * num_steps) for step in [0.1, 0.3, 0.5]])
        scale = 3e-1
        

        D_w = self.noise_w

        if self.dataset == "cifar10":
            D_w = D_w * 10
        
        # Define Discriminator, ways to suppress D: reduce_learning rate, increase label_smooth, enable dropout, reduce Resblock_repeat
        label_smoothing = 0.1
        gan_discriminator = architectures.discriminator((3, 32, 32), True, resblock_repeat = 0, dropout = True)
        optimizer_C = torch.optim.Adam(gan_discriminator.parameters(), lr=lr_C )
        scheduler_C = torch.optim.lr_scheduler.MultiStepLR(optimizer_C, steps, scale)
        
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr_G )
        scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, steps, scale)
        
        train_output_path = self.save_dir + "generator_train"
        if os.path.isdir(train_output_path):
            rmtree(train_output_path)
        os.makedirs(train_output_path)
        # best_acc = 0
        # acc_list = []
        if resume:
            G_state_dict = torch.load(self.save_dir + f"/checkpoint_generator_200.tar")
            self.generator.load_state_dict(G_state_dict)
            self.generator.cuda()
            self.generator.eval()

            z = torch.randn((10, nz)).cuda()
            for i in range(self.num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                #Get fake image from generator
                fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

                imgGen = fake.clone()
                if DENORMALIZE_OPTION:
                    imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                    os.mkdir(train_output_path + "/{}".format(num_steps))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))
        else:
            self.generator.cuda()
            self.generator.train()
            gan_discriminator.cuda()
            gan_discriminator.train()

            if data_helper is not None and discriminator_option:
                data_iterator = iter(data_helper)
            else:
                data_iterator = None

            criterion = torch.nn.CrossEntropyLoss()
            BCE_loss = torch.nn.BCELoss()
            
            bc_losses = AverageMeter()
            bc_losses_gan = AverageMeter()
            ce_losses = AverageMeter()
            g_losses = AverageMeter()



            max_acc = 0
            max_fidelity = 0

            

            for i in range(1, num_steps + 1):
                

                if pred_option:
                    if i % 10 == 0:
                        self.suro_scheduler.step()
                        val_accu, fidel_score = self.steal_test()
                        if val_accu > max_acc:
                            max_acc = val_accu
                        if fidel_score > max_fidelity:
                            max_fidelity = fidel_score
                        self.logger.debug("Step: {}, val_acc: {}, val_fidelity: {}".format(i, val_accu, fidel_score))
                    

                if i % 10 == 0:
                    bc_losses = AverageMeter()
                    bc_losses_gan = AverageMeter()
                    ce_losses = AverageMeter()
                    g_losses = AverageMeter()
                    

                
                scheduler_G.step()
                scheduler_C.step()

                #Sample Random Noise
                z = torch.randn((100, nz)).cuda()
                
                B = 50

                labels_l = torch.randint(low=0, high=self.num_class, size = [B, ]).cuda()
                labels_r = copy.deepcopy(labels_l).cuda()
                labels = torch.stack([labels_l, labels_r]).view(-1)
                zero_label = torch.zeros((50, )).cuda() 
                
                one_label = torch.ones((50, )).cuda() 

                '''Train Generator'''
                optimizer_G.zero_grad()
                
                #Get fake image from generator
                fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation
                
                if i % 10 == 0:
                    imgGen = fake.clone()
                    if DENORMALIZE_OPTION:
                        imgGen = denormalize(imgGen, self.dataset)
                    if not os.path.isdir(train_output_path + "/train"):
                        os.mkdir(train_output_path + "/train")
                    torchvision.utils.save_image(imgGen, train_output_path + '/train/out_{}.jpg'.format(i * 100 + 100))
                
                
                # with torch.no_grad(): 
                output = self.model.local_list[0](fake)

                output = self.model.cloud(output)

                s_output = self.surrogate_model.local(fake)

                s_output = self.surrogate_model.cloud(s_output)
                
                # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
                
                g_noise_out_dist = torch.mean(torch.abs(fake[:B, :] - fake[B:, :]))
                g_noise_z_dist = torch.mean(torch.abs(z[:B, :] - z[B:, :]).view(B,-1),dim=1)
                g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * self.noise_w


                if not pred_option:
                    #Cross Entropy Loss
                    ce_loss = criterion(output, labels)

                    loss = ce_loss - g_noise
                    
                else:
                    # ce_loss = criterion(output, labels) - 0.5* student_Loss(s_output, output)
                    # ce_loss = - 1* student_Loss(s_output, output)
                    ce_loss = criterion(output, labels)
                    # ce_loss = - 0.1* student_Loss(s_output, output)
                    # F.kl_div(s_output, output.detach(), reduction='none').sum(dim = 1).view(-1, m + 1) 
                    loss = ce_loss - g_noise
                ## Discriminator Loss
                if data_helper is not None and discriminator_option:
                    d_out = gan_discriminator(fake)
                    bc_loss_gan = D_w * BCE_loss(d_out.reshape(-1), one_label)
                    loss += bc_loss_gan
                    bc_losses_gan.update(bc_loss_gan.item(), 100)
                
                loss.backward()

                optimizer_G.step()

                ce_losses.update(ce_loss.item(), 100)
                
                g_losses.update(g_noise.item(), 100)

                '''Train surrogate model (to match teacher and student, assuming having prediction access)'''
                if pred_option:
                    
                    for _ in range(d_iter):
                        z = torch.randn((100, nz)).cuda()
                        labels = torch.randint(low=0, high=self.num_class, size = [100, ]).cuda()
                        fake = self.generator(z, labels).detach()
                        

                        with torch.no_grad(): 
                            t_out = self.model.local_list[0](fake)
                            t_out = self.model.cloud(t_out)
                            t_out = F.log_softmax(t_out, dim=1).detach()
                        
                        self.suro_optimizer.zero_grad()
                        s_out = self.surrogate_model.local(fake)
                        s_out = self.surrogate_model.cloud(s_out)

                        loss_S = student_Loss(s_out, t_out) 
                        # loss_S = student_Loss(s_out, t_out) + criterion(s_out, labels)
                        loss_S.backward()
                        self.suro_optimizer.step()

                '''Train Discriminator (tell real/fake, using data_helper)'''
                if data_helper is not None and discriminator_option:
                    try:
                        images, _ = next(data_iterator)
                        if images.size(0) != 100:
                            data_iterator = iter(data_helper)
                            images, _ = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(data_helper)
                        images, _ = next(data_iterator)
                    
                    images = images.cuda()

                    d_input = torch.cat((fake.detach(), images), dim = 0)

                    d_label =  torch.cat((zero_label, one_label - label_smoothing), dim = 0)

                    optimizer_C.zero_grad()

                    d_output = gan_discriminator(d_input)
                    
                    bc_loss = BCE_loss(d_output.reshape(-1), d_label)
                    
                    bc_loss.backward()
                    
                    optimizer_C.step()

                    bc_losses.update(bc_loss.item(), 100)

                # Log Results
                if i % 10 == 0:
                    self.logger.debug(f'Train step: {i}\t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')
            
            if pred_option:
                self.logger.debug("Best perform model, val_acc: {}, fidel_score: {}".format(max_acc, max_fidelity))
            self.logger.debug(f'End of Training: {i}\t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')

            self.generator.cuda()
            self.generator.eval()

            z = torch.randn((10, nz)).cuda()
            for i in range(self.num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                #Get fake image from generator
                fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

                imgGen = fake.clone()
                if DENORMALIZE_OPTION:
                    imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                    os.mkdir(train_output_path + "/{}".format(num_steps))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))

    def steal_test(self, attack_client = 1):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        fidel_score = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        # self.surrogate_model.local.cuda()
        # self.surrogate_model.local.eval()
        # self.surrogate_model.cloud.cuda()
        # self.surrogate_model.cloud.eval()
        # self.surrogate_model.classifier.cuda()
        # self.surrogate_model.classifier.eval()

        self.model.local_list[0].cuda()
        self.model.local_list[0].eval()
        self.model.cloud.cuda()
        self.model.cloud.eval()
        self.classifier.cuda()
        self.classifier.eval()
        

        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            with torch.no_grad():

                output = self.surrogate_model.local(input)
                output_target = self.model.local_list[0](input)

                output = self.surrogate_model.cloud(output)
                output_target = self.model.cloud(output_target)

                if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                    if not self.surrogate_model.cloud_classifier_merge:
                        output = F.avg_pool2d(output, 4)
                        output = output.view(output.size(0), -1)
                        output = self.surrogate_model.classifier(output)
                    if not self.model.cloud_classifier_merge:
                        output_target = F.avg_pool2d(output_target, 4)
                        output_target = output_target.view(output_target.size(0), -1)
                        output_target = self.classifier(output_target)
                elif self.arch == "resnet20" or self.arch == "resnet32":
                    if not self.surrogate_model.cloud_classifier_merge:
                        output = F.avg_pool2d(output, 8)
                        output = output.view(output.size(0), -1)
                        output = self.surrogate_model.classifier(output)
                    if not self.model.cloud_classifier_merge:
                        output_target = F.avg_pool2d(output_target, 8)
                        output_target = output_target.view(output_target.size(0), -1)
                        output_target = self.classifier(output_target)
                else:
                    if not self.surrogate_model.cloud_classifier_merge:
                        output = output.view(output.size(0), -1)
                        output = self.surrogate_model.classifier(output)
                    if not self.model.cloud_classifier_merge:
                        output_target = output_target.view(output_target.size(0), -1)
                        output_target = self.classifier(output_target)

            output = output.float()
            output_target = output_target.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            #TODO: temporary
            # prec_target = accuracy(output_target.data, target)[0]

            fidel = fidelity(output.data, output_target.data)[0]
            top1.update(prec1.item(), input.size(0))
            fidel_score.update(fidel.item(), input.size(0))
        return top1.avg, fidel_score.avg

    def MIA_attack(self, num_epochs, attack_option="MIA", collude_client=1, target_client=0, noise_aware=False,
                   loss_type="MSE", attack_from_later_layer=-1, MIA_optimizer = "Adam", MIA_lr = 1e-3):
        attack_option = attack_option
        MIA_optimizer = MIA_optimizer
        MIA_lr = MIA_lr
        attack_batchsize = 32
        attack_num_epochs = num_epochs
        model_log_file = self.save_dir + '/{}_attack_{}_{}.log'.format(attack_option, collude_client, target_client)
        logger = setup_logger('{}_{}to{}_attack_logger'.format(str(self.save_dir), collude_client, target_client),
                              model_log_file, level=logging.DEBUG)
        # pass
        image_data_dir = self.save_dir + "/img"
        tensor_data_dir = self.save_dir + "/img"

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)

        if self.dataset == "cifar100":
            val_single_loader, _, _ = get_cifar100_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "cifar10":
            val_single_loader, _, _ = get_cifar10_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "svhn":
            val_single_loader, _, _ = get_SVHN_testloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "mnist":
            _, val_single_loader = get_mnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "fmnist":
            _, val_single_loader = get_fmnist_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "facescrub":
            _, val_single_loader = get_facescrub_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "tinyimagenet":
            _, val_single_loader = get_tinyimagenet_bothloader(batch_size=1, num_workers=4, shuffle=False)
        elif self.dataset == "imagenet":
            val_single_loader = get_imagenet_testloader(batch_size=1, num_workers=4, shuffle=False)
        attack_path = self.save_dir + '/{}_attack_{}to{}'.format(attack_option, collude_client, target_client)
        if not os.path.isdir(attack_path):
            os.makedirs(attack_path)
            os.makedirs(attack_path + "/train")
            os.makedirs(attack_path + "/test")
            os.makedirs(attack_path + "/tensorboard")
            os.makedirs(attack_path + "/sourcecode")
        train_output_path = "{}/train".format(attack_path)
        test_output_path = "{}/test".format(attack_path)
        tensorboard_path = "{}/tensorboard/".format(attack_path)
        model_path = "{}/model.pt".format(attack_path)
        path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                     "test_output_path": test_output_path, "tensorboard_path": tensorboard_path}

        if ("MIA" in attack_option) and ("MIA_mf" not in attack_option):
            logger.debug("Generating IR ...... (may take a while)")

            if collude_client == 0:
                self.gen_ir(val_single_loader, self.f, image_data_dir, tensor_data_dir,
                            attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
            elif collude_client == 1:
                self.gen_ir(val_single_loader, self.c, image_data_dir, tensor_data_dir,
                            attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
            elif collude_client > 1:
                self.gen_ir(val_single_loader, self.model.local_list[collude_client], image_data_dir, tensor_data_dir,
                            attack_from_later_layer=attack_from_later_layer, attack_option = attack_option)
            for filename in os.listdir(tensor_data_dir):
                if ".pt" in filename:
                    sampled_tensor = torch.load(tensor_data_dir + "/" + filename)
                    input_nc = sampled_tensor.size()[1]
                    try:
                        input_dim = sampled_tensor.size()[2]
                    except:
                        print("Extract input dimension fialed, set to 0")
                        input_dim = 0
                    break

            if self.gan_AE_type == "custom":
                decoder = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                  activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "custom_bn":
                decoder = architectures.custom_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                     activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex":
                decoder = architectures.complex_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                   activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_plus":
                decoder = architectures.complex_plus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_res":
                decoder = architectures.complex_res_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                       output_dim=32, activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "complex_resplus":
                decoder = architectures.complex_resplus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                           output_dim=32, activation=self.gan_AE_activation).cuda()
            elif "complex_resplusN" in self.gan_AE_type:
                try:
                    N = int(self.gan_AE_type.split("complex_resplusN")[1])
                except:
                    print("auto extract N from complex_resplusN failed, set N to default 2")
                    N = 2
                decoder = architectures.complex_resplusN_AE(N = N, input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                               output_dim=32, activation=self.gan_AE_activation).cuda()
            elif "complex_normplusN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("complex_normplusN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from complex_normplusN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            
            elif "conv_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("conv_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from conv_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()

            elif "res_normN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("res_normN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from res_normN failed, set N to default 2")
                    N = 0
                    internal_C = 64
                decoder = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            
            elif "TB_normplusN" in self.gan_AE_type:
                try:
                    afterfix = self.gan_AE_type.split("TB_normplusN")[1]
                    N = int(afterfix.split("C")[0])
                    internal_C = int(afterfix.split("C")[1])
                except:
                    print("auto extract N from TB_normplusN failed, set N to default 0")
                    N = 0
                    internal_C = 64
                decoder = architectures.TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                            input_dim=input_dim, output_dim=32,
                                                            activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simple":
                decoder = architectures.simple_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                  activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simple_bn":
                decoder = architectures.simple_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                     activation=self.gan_AE_activation).cuda()
            elif self.gan_AE_type == "simplest":
                decoder = architectures.simplest_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim, output_dim=32,
                                                    activation=self.gan_AE_activation).cuda()
            else:
                raise ("No such GAN AE type.")

            if self.measure_option:
                noise_input = torch.randn([1, input_nc, input_dim, input_dim])
                device = next(decoder.parameters()).device
                noise_input = noise_input.to(device)
                macs, num_param = profile(decoder, inputs=(noise_input,))
                self.logger.debug(
                    "{} Decoder Model's Mac and Param are {} and {}".format(self.gan_AE_type, macs, num_param))
                '''Uncomment below to also get decoder's inference and training time overhead.'''
                # decoder.cpu()
                # noise_input = torch.randn([128, input_nc, input_dim, input_dim])
                # with torch.no_grad():
                #     _ = decoder(noise_input)
                #     start_time = time.time()
                #     for _ in range(500):  # CPU warm up
                #         _ = decoder(noise_input)
                #     lapse_cpu_decoder = (time.time() - start_time) / 500
                # self.logger.debug("Decoder Model's Inference time on CPU is {}".format(lapse_cpu_decoder))

                # criterion = torch.nn.MSELoss()
                # noise_reconstruction = torch.randn([128, 3, 32, 32])
                # reconstruction = decoder(noise_input)

                # r_loss = criterion(reconstruction, noise_reconstruction)
                # r_loss.backward()
                # lapse_cpu_decoder_train = 0
                # for _ in range(500):  # CPU warm up
                #     reconstruction = decoder(noise_input)
                #     r_loss = criterion(reconstruction, noise_reconstruction)
                #     start_time = time.time()
                #     r_loss.backward()
                #     lapse_cpu_decoder_train += (time.time() - start_time)
                # lapse_cpu_decoder_train = lapse_cpu_decoder_train / 500
                # del r_loss, reconstruction, noise_input
                # self.logger.debug("Decoder Model's Train time on CPU is {}".format(lapse_cpu_decoder_train))
                # decoder.cuda()

            '''Setting attacker's learning algorithm'''
            # optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            if MIA_optimizer == "Adam":
                optimizer = torch.optim.Adam(decoder.parameters(), lr=MIA_lr)
            elif MIA_optimizer == "SGD":
                optimizer = torch.optim.SGD(decoder.parameters(), lr=MIA_lr)
            else:
                raise("MIA optimizer {} is not supported!".format(MIA_optimizer))
            # Construct a dataset for training the decoder
            trainloader, testloader = apply_transform(attack_batchsize, image_data_dir, tensor_data_dir)

            # Do real test on target's client activation (and test with target's client ground-truth.)
            sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                          self.n_epochs))

            if "gan_adv_noise" in self.regularization_option and noise_aware:
                print("create a second decoder") # Avoid using the same decoder as the inference user uses [see "def save_image_act_pair"].
                if self.gan_AE_type == "custom":
                    decoder2 = architectures.custom_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                       output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "custom_bn":
                    decoder2 = architectures.custom_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                          output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "complex":
                    decoder2 = architectures.complex_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                        output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "complex_plus":
                    decoder2 = architectures.complex_plus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                             output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "complex_res":
                    decoder2 = architectures.complex_res_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                            output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "complex_resplus":
                    decoder2 = architectures.complex_resplus_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                                output_dim=32, activation=self.gan_AE_activation).cuda()
                elif "complex_resplusN" in self.gan_AE_type:
                    try:
                        N = int(self.gan_AE_type.split("complex_resplusN")[1])
                    except:
                        print("auto extract N from complex_resplusN failed, set N to default 2")
                        N = 2
                    decoder2 = architectures.complex_resplusN_AE(N = N, input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                                output_dim=32, activation=self.gan_AE_activation).cuda()
                elif "complex_normplusN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("complex_normplusN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from complex_normplusN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.complex_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()

                elif "conv_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("conv_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from conv_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.conv_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()
                
                elif "res_normN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("res_normN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from res_normN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.res_normN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()

                elif "TB_normplusN" in self.gan_AE_type:
                    try:
                        afterfix = self.gan_AE_type.split("TB_normplusN")[1]
                        N = int(afterfix.split("C")[0])
                        internal_C = int(afterfix.split("C")[1])
                    except:
                        print("auto extract N from TB_normplusN failed, set N to default 2")
                        N = 0
                        internal_C = 64
                    decoder2 = architectures.TB_normplusN_AE(N = N, internal_nc = internal_C, input_nc=input_nc, output_nc=3,
                                                                input_dim=input_dim, output_dim=32,
                                                                activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "simple":
                    decoder2 = architectures.simple_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                       output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "simple_bn":
                    decoder2 = architectures.simple_AE_bn(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                          output_dim=32, activation=self.gan_AE_activation).cuda()
                elif self.gan_AE_type == "simplest":
                    decoder2 = architectures.simplest_AE(input_nc=input_nc, output_nc=3, input_dim=input_dim,
                                                         output_dim=32, activation=self.gan_AE_activation).cuda()
                else:
                    raise ("No such GAN AE type.")
                # optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
                self.attack(attack_num_epochs, decoder2, optimizer2, trainloader, testloader, logger, path_dict,
                            attack_batchsize, pretrained_decoder=self.local_AE_list[collude_client], noise_aware=noise_aware)
                decoder = decoder2  # use decoder2 for testing
            else:
                # Perform Input Extraction Attack
                self.attack(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                            attack_batchsize, noise_aware=noise_aware, loss_type=loss_type)

            
            # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
            mse_score, ssim_score, psnr_score = self.test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                 path_dict, attack_batchsize,
                                                                 num_classes=self.num_class)

            # Clear content of image_data_dir/tensor_data_dir
            if os.path.isdir(image_data_dir):
                rmtree(image_data_dir)
            if os.path.isdir(tensor_data_dir):
                rmtree(tensor_data_dir)
            return mse_score, ssim_score, psnr_score
        elif attack_option == "MIA_mf":  # Stands for Model-free MIA, does not need a AE model, optimize each fake image instead.

            lambda_TV = 0.0
            lambda_l2 = 0.0
            num_step = attack_num_epochs * 60

            sp_testloader = apply_transform_test(1, self.save_dir + "/save_activation_client_{}_epoch_{}".format(
                target_client, self.n_epochs), self.save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                          self.n_epochs))
            criterion = nn.MSELoss().cuda()
            ssim_loss = pytorch_ssim.SSIM()
            all_test_losses = AverageMeter()
            ssim_test_losses = AverageMeter()
            psnr_test_losses = AverageMeter()
            fresh_option = True
            for num, data in enumerate(sp_testloader, 1):
                # img, ir, _ = data
                img, ir, _ = data

                # optimize a fake_image to (1) have similar ir, (2) have small total variance, (3) have small l2
                img = img.cuda()
                if not fresh_option:
                    ir = ir.cuda()
                self.model.local_list[collude_client].eval()
                self.model.local_list[target_client].eval()

                fake_image = torch.zeros(img.size(), requires_grad=True, device="cuda")
                optimizer = torch.optim.Adam(params=[fake_image], lr=8e-1, amsgrad=True, eps=1e-3)
                # optimizer = torch.optim.Adam(params = [fake_image], lr = 1e-2, amsgrad=True, eps=1e-3)
                for step in range(1, num_step + 1):
                    optimizer.zero_grad()

                    fake_ir = self.model.local_list[collude_client](fake_image)  # Simulate Original

                    if fresh_option:
                        ir = self.model.local_list[target_client](img)  # Getting fresh ir from target local model

                    featureLoss = criterion(fake_ir, ir)

                    TVLoss = TV(fake_image)
                    normLoss = l2loss(fake_image)

                    totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

                    totalLoss.backward()

                    optimizer.step()
                    # if step % 100 == 0:
                    if step == 0 or step == num_step:
                        logger.debug("Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(step,
                                                                                             featureLoss.cpu().detach().numpy(),
                                                                                             TVLoss.cpu().detach().numpy(),
                                                                                             normLoss.cpu().detach().numpy()))
                imgGen = fake_image.clone()
                imgOrig = img.clone()

                mse_loss = criterion(imgGen, imgOrig)
                ssim_loss_val = ssim_loss(imgGen, imgOrig)
                psnr_loss_val = get_PSNR(imgOrig, imgGen)
                all_test_losses.update(mse_loss.item(), ir.size(0))
                ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
                psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
                if not os.path.isdir(test_output_path + "/{}".format(attack_num_epochs)):
                    os.mkdir(test_output_path + "/{}".format(attack_num_epochs))
                if DENORMALIZE_OPTION:
                    imgGen = denormalize(imgGen, self.dataset)
                torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(attack_num_epochs,
                                                                                                 num * attack_batchsize + attack_batchsize))
                if DENORMALIZE_OPTION:
                    imgOrig = denormalize(imgOrig, self.dataset)
                torchvision.utils.save_image(imgOrig, test_output_path + '/{}/inp_{}.jpg'.format(attack_num_epochs,
                                                                                                 num * attack_batchsize + attack_batchsize))
                # imgGen = deprocess(imgGen, self.num_class)
                # imgOrig = deprocess(imgOrig, self.num_class)
                # torchvision.utils.save_image(imgGen, test_output_path + '/{}/dp_out_{}.jpg'.format(attack_num_epochs, num*attack_batchsize + attack_batchsize))
                # torchvision.utils.save_image(imgOrig, test_output_path + '/{}/dp_inp_{}.jpg'.format(attack_num_epochs, num*attack_batchsize + attack_batchsize))
            logger.debug("MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                all_test_losses.avg))
            logger.debug("SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                ssim_test_losses.avg))
            logger.debug("PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
                psnr_test_losses.avg))
            return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    def attack(self, num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict, batch_size,
               loss_type="MSE", pretrained_decoder=None, noise_aware=False):
        round_ = 0
        min_val_loss = 999.
        max_val_loss = 0.
        train_output_freq = 10
        # test_output_freq = 50
        train_losses = AverageMeter()
        val_losses = AverageMeter()

        # Optimize based on MSE distance
        if loss_type == "MSE":
            criterion = nn.MSELoss()
        elif loss_type == "SSIM":
            criterion = pytorch_ssim.SSIM()
        elif loss_type == "PSNR":
            criterion = None
        else:
            raise ("No such loss in self.attack")
        device = next(decoder.parameters()).device
        decoder.train()
        for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
            for num, data in enumerate(trainloader, 1):
                img, ir = data
                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)
                # print(img)
                # Use local DP for training the AE.
                if self.local_DP and noise_aware:
                    with torch.no_grad():
                        if "laplace" in self.regularization_option:
                            ir += torch.from_numpy(
                                np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=ir.size())).cuda()
                        else:  # apply gaussian noise
                            delta = 10e-5
                            sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                            ir += sigma * torch.randn_like(ir).cuda()
                if self.dropout_defense and noise_aware:
                    ir = dropout_defense(ir, self.dropout_ratio)
                if self.topkprune and noise_aware:
                    ir = prune_defense(ir, self.topkprune_ratio)
                if pretrained_decoder is not None and "gan_adv_noise" in self.regularization_option and noise_aware:
                    epsilon = self.alpha2
                    
                    pretrained_decoder.eval()
                    fake_act = ir.clone()
                    grad = torch.zeros_like(ir).cuda()
                    fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                    x_recon = pretrained_decoder(fake_act)
                    if self.gan_loss_type == "SSIM":
                        ssim_loss = pytorch_ssim.SSIM()
                        loss = ssim_loss(x_recon, img)
                        loss.backward()
                        grad -= torch.sign(fake_act.grad)
                    else:
                        mse_loss = nn.MSELoss()
                        loss = mse_loss(x_recon, img)
                        loss.backward()
                        grad += torch.sign(fake_act.grad)
                    # ir = ir + grad.detach() * epsilon
                    ir = ir - grad.detach() * epsilon
                # print(ir.size())
                output = decoder(ir)

                if loss_type == "MSE":
                    reconstruction_loss = criterion(output, img)
                elif loss_type == "SSIM":
                    reconstruction_loss = -criterion(output, img)
                elif loss_type == "PSNR":
                    reconstruction_loss = -1 / 10 * get_PSNR(img, output)
                else:
                    raise ("No such loss in self.attack")
                train_loss = reconstruction_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                train_losses.update(train_loss.item(), ir.size(0))

            if (epoch + 1) % train_output_freq == 0:
                save_images(img, output, epoch, path_dict["train_output_path"], offset=0, batch_size=batch_size)

            for num, data in enumerate(testloader, 1):
                img, ir = data

                img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
                img, ir = Variable(img).to(device), Variable(ir).to(device)

                output = decoder(ir)

                reconstruction_loss = criterion(output, img)
                val_loss = reconstruction_loss

                if loss_type == "MSE" and val_loss < min_val_loss:
                    min_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "SSIM" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                elif loss_type == "PSNR" and val_loss > max_val_loss:
                    max_val_loss = val_loss
                    torch.save(decoder.state_dict(), path_dict["model_path"])
                val_losses.update(val_loss.item(), ir.size(0))

                self.writer.add_scalar('decoder_loss/val', val_loss.item(), len(testloader) * epoch + num)
                self.writer.add_scalar('decoder_loss/val_loss/reconstruction', reconstruction_loss.item(),
                                       len(testloader) * epoch + num)

            for name, param in decoder.named_parameters():
                self.writer.add_histogram("decoder_params/{}".format(name), param.clone().cpu().data.numpy(), epoch)

            # torch.save(decoder.state_dict(), path_dict["model_path"])
            logger.debug(
                "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                    epoch + 1,
                    num_epochs, train_losses=train_losses, val_losses=val_losses))
        if loss_type == "MSE":
            logger.debug("Best Validation Loss is {}".format(min_val_loss))
        elif loss_type == "SSIM":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))
        elif loss_type == "PSNR":
            logger.debug("Best Validation Loss is {}".format(max_val_loss))

    def test_attack(self, num_epochs, decoder, sp_testloader, logger, path_dict, batch_size, num_classes=10,
                    select_label=0):
        device = next(decoder.parameters()).device
        # print("Load the best Decoder Model...")
        new_state_dict = torch.load(path_dict["model_path"])
        decoder.load_state_dict(new_state_dict)
        decoder.eval()
        # test_losses = []
        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()
        ssim_loss = pytorch_ssim.SSIM()

        criterion = nn.MSELoss()

        for num, data in enumerate(sp_testloader, 1):
            img, ir, label = data

            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)
            output_imgs = decoder(ir)
            reconstruction_loss = criterion(output_imgs, img)
            ssim_loss_val = ssim_loss(output_imgs, img)
            psnr_loss_val = get_PSNR(img, output_imgs)
            all_test_losses.update(reconstruction_loss.item(), ir.size(0))
            ssim_test_losses.update(ssim_loss_val.item(), ir.size(0))
            psnr_test_losses.update(psnr_loss_val.item(), ir.size(0))
            save_images(img, output_imgs, num_epochs, path_dict["test_output_path"], offset=num, batch_size=batch_size)

        logger.debug(
            "MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(all_test_losses.avg))
        logger.debug(
            "SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(ssim_test_losses.avg))
        logger.debug(
            "PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(psnr_test_losses.avg))
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

    def save_activation_bhtsne(self, save_activation, labels, batch_size, msg1, msg2):
        """
            Run one train epoch
        """

        save_activation = save_activation.float()
        save_activation = save_activation.detach().cpu().numpy()
        save_activation = save_activation.reshape(batch_size, -1)
        f=open(os.path.join(self.save_dir, "{}_{}_act.txt".format(msg1, msg2)),'a')
        np.savetxt(f, save_activation, fmt='%.2f')
        f.close()

        target = labels.cpu().numpy()
        target = target.reshape(batch_size, -1)
        f=open(os.path.join(self.save_dir, "{}_{}_target.txt".format(msg1, msg2)),'a')
        np.savetxt(f, target, fmt='%d')
        f.close()

    #Generate test set for MIA decoder
    def save_image_act_pair(self, input, target, client_id, epoch, clean_option=False, attack_from_later_layer=-1, attack_option = "MIA"):
        """
            Run one train epoch
        """
        path_dir = os.path.join(self.save_dir, 'save_activation_client_{}_epoch_{}'.format(client_id, epoch))
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        else:
            rmtree(path_dir)
            os.makedirs(path_dir)
        input = input.cuda()

        for j in range(input.size(0)):
            img = input[None, j, :, :, :]
            label = target[None, j]
            with torch.no_grad():
                if client_id == 0:
                    self.f.eval()
                    save_activation = self.f(img)
                elif client_id == 1:
                    self.c.eval()
                    save_activation = self.c(img)
                elif client_id > 1:
                    self.model.local_list[client_id].eval()
                    save_activation = self.model.local_list[client_id](img)
                if self.confidence_score:
                    self.model.cloud.eval()
                    save_activation = self.model.cloud(save_activation)
                    if self.arch == "resnet18" or self.arch == "resnet34" or "mobilenetv2" in self.arch:
                        save_activation = F.avg_pool2d(save_activation, 4)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    elif self.arch == "resnet20" or self.arch == "resnet32":
                        save_activation = F.avg_pool2d(save_activation, 8)
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
                    else:
                        save_activation = save_activation.view(save_activation.size(0), -1)
                        save_activation = self.classifier(save_activation)
            

            if attack_from_later_layer > -1 and (not self.confidence_score):
                self.model.cloud.eval()

                activation_3 = {}

                def get_activation_3(name):
                    def hook(model, input, output):
                        activation_3[name] = output.detach()

                    return hook

                with torch.no_grad():
                    activation_3 = {}
                    count = 0
                    for name, m in self.model.cloud.named_modules():
                        if attack_from_later_layer == count:
                            m.register_forward_hook(get_activation_3("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = self.model.cloud(save_activation)
                try:
                    save_activation = activation_3[valid_key]
                except:
                    print("cannot attack from later layer, server-side model is empty or does not have enough layers")
            
            if self.local_DP and not clean_option:  # local DP or additive noise
                if "laplace" in self.regularization_option:
                    save_activation += torch.from_numpy(
                        np.random.laplace(loc=0, scale=1 / self.dp_epsilon, size=save_activation.size())).cuda()
                    # the addtive work uses scale in (0.1 0.5 1.0) -> (1 2 10) regularization_strength (self.dp_epsilon)
                else:  # apply gaussian noise
                    delta = 10e-5
                    sigma = np.sqrt(2 * np.log(1.25 / delta)) * 1 / self.dp_epsilon
                    save_activation += sigma * torch.randn_like(save_activation).cuda()
            if self.dropout_defense and not clean_option:  # activation dropout defense
                save_activation = dropout_defense(save_activation, self.dropout_ratio)
            if self.topkprune and not clean_option:
                save_activation = prune_defense(save_activation, self.topkprune_ratio)
            if DENORMALIZE_OPTION:
                img = denormalize(img, self.dataset)
                
            if self.gan_noise and not clean_option:
                epsilon = self.alpha2
                self.local_AE_list[client_id].eval()
                fake_act = save_activation.clone()
                grad = torch.zeros_like(save_activation).cuda()
                fake_act = torch.autograd.Variable(fake_act.cuda(), requires_grad=True)
                x_recon = self.local_AE_list[client_id](fake_act)
                
                if self.gan_loss_type == "SSIM":
                    ssim_loss = pytorch_ssim.SSIM()
                    loss = ssim_loss(x_recon, img)
                    loss.backward()
                    grad -= torch.sign(fake_act.grad)
                elif self.gan_loss_type == "MSE":
                    mse_loss = torch.nn.MSELoss()
                    loss = mse_loss(x_recon, img)
                    loss.backward()
                    grad += torch.sign(fake_act.grad)  

                save_activation = save_activation - grad.detach() * epsilon
            if "truncate" in attack_option:
                save_activation = prune_top_n_percent_left(save_activation)
            
            save_activation = save_activation.float()
            
            save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
            torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
            torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))

# if __name__ == '__main__':
#     a = [torch.ones([128, 8]), torch.ones([128, 8]), torch.ones([128, 8]), torch.ones([128, 8])]
#     max_grad, max_idx = label_deduction(a)
#     print(max_idx)
#     print(max_grad.size())
#     print(max_idx.size())