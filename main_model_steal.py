# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import MIA_torch
from datasets_torch import *
from utils import setup_logger
import argparse

import torch



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# match below exactlly the same with the training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=4, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', required=True, type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=1, type=int, help='number of client')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=0.05, type=float, help='Learning Rate for server-side model')
parser.add_argument('--test_best', action='store_true', default=False, help='if True, test the best epoch')

parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--random_seed', default=123, type=int, help='random_seed for the testing dataset')
parser.add_argument('--scheme', default="V2_epoch", type=str, help='the name of the scheme, either V3 or others')
parser.add_argument('--bottleneck_option', default="None", type=str, help='set bottleneck option')
parser.add_argument('--adversairal_attack', action='store_true', default=False, help='if True, test transfer adversarial attack')
# test setting
parser.add_argument('--regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0.0, type=float, help='regularization_strength of regularization in multi-client training.')
parser.add_argument('--average_time', default=1, type=int, help='number of time to run the MIA attack for an average performance')

# steal setting
parser.add_argument('--attack_client', default=1, type=int, help='id of the thief client')
parser.add_argument('--attack_epochs', default=50, type=int, help='number of epochs for the steal algorithm')
parser.add_argument('--num_query', default=50, type=int, help='number of query for the steal algorithm')
parser.add_argument('--surrogate_arch', default="same", type=str, help='set surrogate_arch')
parser.add_argument('--attack_style', default="None", type=str, help='set attack_style')
parser.add_argument('--data_proportion', default=0.2, type=float, help='data_proportion in multi-client training.')
parser.add_argument('--noniid_ratio', default=1.0, type=float, help='noniid_ratio')
parser.add_argument('--train_clas_layer', default=3, type=float, help='train_clas_layer.')
args = parser.parse_args()

batch_size = args.batch_size
cutting_layer = args.cutlayer
date_list = []
date_list.append(args.filename)
num_client = args.num_client

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

for date_0 in date_list:

    if args.test_best:
        args.num_epochs = "best"
    save_dir_name = "./{}/{}".format(args.folder, date_0)
    mi = MIA_torch.MIA(args.arch, cutting_layer, batch_size, n_epochs = args.num_epochs, scheme = args.scheme, 
                        num_client = num_client, dataset=args.dataset, save_dir= save_dir_name,  learning_rate = args.learning_rate,
                        regularization_option=args.regularization, regularization_strength = args.regularization_strength, 
                        random_seed = args.random_seed, bottleneck_option = args.bottleneck_option)

    new_folder_dir = mi.save_dir + '/layer_{}_data_{}_stealtype_{}_query{}_epoch{}_surrogate_{}/'.format(args.train_clas_layer, args.data_proportion, args.attack_style, args.num_query, args.attack_epochs, args.surrogate_arch)
    if args.noniid_ratio < 1.0:
        new_folder_dir = mi.save_dir + '/layer_{}_data_{}_stealtype_{}_query{}_epoch{}_surrogate_{}_noniid/'.format(args.train_clas_layer, args.data_proportion, args.attack_style, args.num_query, args.attack_epochs, args.surrogate_arch)
    new_folder_dir = os.path.abspath(new_folder_dir)
    if not os.path.isdir(new_folder_dir):
        os.makedirs(new_folder_dir)
    model_log_file = new_folder_dir + '/MIA.log'
    mi.logger = setup_logger('{}_logger'.format(str(save_dir_name)), model_log_file, level=logging.DEBUG)

    if "orig" not in args.scheme:
        mi.resume("./{}/{}/checkpoint_f_{}.tar".format(args.folder, date_0, args.num_epochs))
    else:
        print("resume orig scheme's checkpoint")
        mi.resume("./{}/{}/checkpoint_{}.tar".format(args.folder, date_0, args.num_epochs))

    log_frequency = 500
    skip_valid = True
    if not skip_valid:
        LOG = mi(verbose=True, progress_bar=True, log_frequency=log_frequency)


    mi.logger.debug(str(args))
    mi.steal_attack(num_query = args.num_query, num_epoch = args.attack_epochs, attack_client=args.attack_client, attack_style = args.attack_style, data_proportion = args.data_proportion,
                     noniid_ratio = args.noniid_ratio, train_clas_layer = args.train_clas_layer, surrogate_arch = args.surrogate_arch, adversairal_attack_option = args.adversairal_attack)
    

# %%
