# %%
'''
This is the script to train SFL

[x] support non-iid, dataset split
[x] support stealing during training
[ ] scale to large #Clients
'''


import torch
import numpy as np
import SFL
import os
import argparse
from attacks.model_extraction_attack import steal_attack
from utils import cutting_train_clas_layer_mapping
parser = argparse.ArgumentParser(description='SFL Training')

# training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=8, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', default="vgg11-cifar10", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves/train-ME", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=10, type=int, help='number of client')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--last_client_fix_amount', default=500, type=int, help='last_client_fix_amount')
parser.add_argument('--learning_rate', default=0.05, type=float, help='Learning Rate')
parser.add_argument('--learning_rate_MEA', default=-1, type=float, help='Learning Rate of MEA, if not set, then set it to be the same as SFL learnign rate')
parser.add_argument('--client_sample_ratio', default=1.0, type=float, help='client_sample_ratio')
parser.add_argument('--noniid_ratio', default=1.0, type=float, help='noniid_ratio, if = 0.1, meaning 1 out of 10 class per client')
parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--scheme', default="V2", type=str, help='the name of the scheme, either V3 or others')

#regularization setting ()
parser.add_argument('--regularization', default="craft_train_ME_start160", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0, type=float, help='regularization_strength of regularization in multi-client training.')

#training dataset setting ()
parser.add_argument('--load_from_checkpoint', action='store_true', default=False, help='if True, we load_from_checkpoint')
parser.add_argument('--load_from_checkpoint_server', action='store_true', default=False, help='if True, we load_from_checkpoint for server-side model')
parser.add_argument('--transfer_source_task', default="cifar100", type=str, help='the name of the transfer_source_task, option: cifar10, cifar100')
parser.add_argument('--finetune_freeze_bn', action='store_true', default=False, help='if True, we finetune_freeze_bn')

#training randomseed setting ()
parser.add_argument('--random_seed', default=123, type=int, help='random_seed')

# steal setting
parser.add_argument('--attack_client', default=0, type=int, help='id of the thief client')
parser.add_argument('--attack_epochs', default=50, type=int, help='number of epochs for the steal algorithm')
parser.add_argument('--last_n_batch', default=10000, type=int, help='use last_n_batch collected input-label-pairs to perform the surrogate model training')
parser.add_argument('--surrogate_arch', default="same", type=str, help='set surrogate_arch')
parser.add_argument('--aux_dataset', default="cifar100", type=str, help='number of classes for the testing dataset')
parser.add_argument('--adversairal_attack', action='store_true', default=False, help='if True, test transfer adversarial attack')

args = parser.parse_args()

random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
batch_size = args.batch_size
cutting_layer = args.cutlayer
num_client = args.num_client
save_dir_name = "./{}/{}".format(args.folder, args.filename)

if args.scheme == "V1":
    if num_client == 100:
        batch_size = 4
    elif num_client == 50:
        batch_size = 8
    elif num_client == 20:
        batch_size = 16
    elif num_client == 10:
        batch_size = 32
    elif num_client == 5:
        batch_size = 64
    elif num_client == 2:
        batch_size = 128

if num_client > 20:
    client_sample_ratio = 20. /num_client
    args.num_epochs = int(200 / client_sample_ratio)
else:
    client_sample_ratio = 1.0

mi = SFL.Trainer(args.arch, cutting_layer, batch_size, n_epochs = args.num_epochs, scheme = args.scheme,
                 num_client = num_client, dataset=args.dataset, save_dir=save_dir_name,
                 regularization_option=args.regularization, regularization_strength = args.regularization_strength, learning_rate = args.learning_rate,
                 load_from_checkpoint = args.load_from_checkpoint, finetune_freeze_bn = args.finetune_freeze_bn, client_sample_ratio = client_sample_ratio,
                 source_task = args.transfer_source_task, load_from_checkpoint_server = args.load_from_checkpoint_server, noniid = args.noniid_ratio, last_client_fix_amount = args.last_client_fix_amount)
mi.logger.debug(str(args))



if not os.path.isfile(f"./{args.folder}/{args.filename}/checkpoint_client_{args.num_epochs}.tar"):
    mi.train(verbose=True)
    mi.resume("./{}/{}/checkpoint_client_{}.tar".format(args.folder, args.filename, args.num_epochs))
else:
    mi.resume("./{}/{}/checkpoint_client_{}.tar".format(args.folder, args.filename, args.num_epochs))


train_clas_layer = mi.model.get_num_of_cloud_layer()

if "craft_train" in args.regularization:
    attack_style = "Craft_option_resume"
elif "GM_train" in args.regularization:
    attack_style = "GM_option_resume"
elif "gan_train" in args.regularization:
    attack_style = "Generator_option_resume"
elif "soft_train" in args.regularization:
    attack_style = "SoftTrain_option_resume"
elif "gan_assist_train" in args.regularization:
    attack_style = "Generator_assist_option_resume"
elif "naive_train" in args.regularization:
    attack_style = "NaiveTrain_option_resume"

if args.learning_rate_MEA == -1:
    args.learning_rate_MEA = args.learning_rate

steal_attack(save_dir_name, args.arch, args.cutlayer, mi.num_class, mi.model, args.dataset, mi.pub_dataloader,
             args.aux_dataset, args.batch_size, args.learning_rate_MEA, 50, args.attack_epochs,
             args.attack_client, attack_style, 1.0, args.noniid_ratio, 
             train_clas_layer, args.surrogate_arch, args.adversairal_attack, args.last_n_batch)


# %%
