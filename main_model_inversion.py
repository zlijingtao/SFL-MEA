# %%
import torch
import numpy as np
import logging
import SFL
from datasets import *
from utils import setup_logger
import argparse
from attacks.model_inversion_attack import MIA_attack
import torch



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# match below exactlly the same with the training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=8, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', default="vgg11-cifar10", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves/baseline", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=2, type=int, help='number of client')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')

parser.add_argument('--random_seed', default=123, type=int, help='random_seed for the testing dataset')
parser.add_argument('--scheme', default="V2", type=str, help='the name of the scheme, either V3 or others')

# test setting
parser.add_argument('--regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0.0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--attack_scheme', default="MIA", type=str, help='the name of the attack scheme, either MIA or MIA_mf')
parser.add_argument('--aux_dataset_name', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--attack_epochs', default=50, type=int, help='number of epochs for the MIA attack algorithm')
parser.add_argument('--select_layer_output', default=-1, type=int, help='select which layers activation to attack')
parser.add_argument('--gan_AE_type', default="res_normN4C64", type=str, help='the name of the AE used in attack, option: custom, simple, simplest')
parser.add_argument('--attack_loss_type', default="MSE", type=str, help='the type of the loss function used in attack, option: MSE, SSIM')
parser.add_argument('--gan_loss_type', default="SSIM", type=str, help='loss type of training defensive decoder: SSIM or MSE')
parser.add_argument('--MIA_optimizer', default="Adam", type=str, help='the type of the learning algorithm used in attack, option: Adam, SGD')
parser.add_argument('--MIA_lr', default=1e-3, type=float, help='learning rate used in attack.')

args = parser.parse_args()

args.num_epochs = "best"

save_dir_name = "./{}/{}".format(args.folder, args.filename)

mi = SFL.Trainer(args.arch, args.cutlayer, args.batch_size, n_epochs = args.num_epochs, scheme = args.scheme,
                num_client = args.num_client, dataset=args.dataset, save_dir=save_dir_name,
                regularization_option=args.regularization, regularization_strength = args.regularization_strength)

mi.resume("./{}/{}/checkpoint_client_{}.tar".format(args.folder, args.filename, args.num_epochs))

random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

mse_score, ssim_score, psnr_score = MIA_attack(save_dir_name, mi.model, target_dataset_name = args.dataset, 
            aux_dataset_name = args.aux_dataset_name, num_epochs = args.attack_epochs, 
            attack_option=args.attack_scheme, target_client=0, gan_AE_type = args.gan_AE_type, 
            loss_type=args.attack_loss_type, select_layer_output=args.select_layer_output, MIA_optimizer = args.MIA_optimizer, MIA_lr = args.MIA_lr)

# %%
