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
import argparse
import wandb

parser = argparse.ArgumentParser(description='SFL Training')

# training setting ()
parser.add_argument('--arch', default="vgg11_bn", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--cutlayer', default=4, type=int, help='number of layers in local')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--filename', default="vgg11-cifar10", type=str, help='please type save_file name for the testing purpose')
parser.add_argument('--folder', default="saves/baseline", type=str, help='please type folder name for the testing purpose')
parser.add_argument('--num_client', default=2, type=int, help='number of client')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--learning_rate', default=0.05, type=float, help='Learning Rate')
parser.add_argument('--client_sample_ratio', default=1.0, type=float, help='client_sample_ratio')
parser.add_argument('--noniid_ratio', default=1.0, type=float, help='noniid_ratio, if = 0.1, meaning 1 out of 10 class per client')
parser.add_argument('--dataset', default="cifar10", type=str, help='number of classes for the testing dataset')
parser.add_argument('--scheme', default="V2", type=str, help='the name of the scheme, either V3 or others')

#regularization setting ()
parser.add_argument('--regularization', default="None", type=str, help='apply regularization in multi-client training.')
parser.add_argument('--regularization_strength', default=0, type=float, help='regularization_strength of regularization in multi-client training.')

#training dataset setting ()
parser.add_argument('--load_from_checkpoint', action='store_true', default=False, help='if True, we load_from_checkpoint')
parser.add_argument('--load_from_checkpoint_server', action='store_true', default=False, help='if True, we load_from_checkpoint for server-side model')
parser.add_argument('--transfer_source_task', default="cifar100", type=str, help='the name of the transfer_source_task, option: cifar10, cifar100')
parser.add_argument('--finetune_freeze_bn', action='store_true', default=False, help='if True, we finetune_freeze_bn')
parser.add_argument('--collude_use_public', action='store_true', default=False, help='if True, we use validation dataset to traing the collude clients (all other client has client_id > 0)')

#training randomseed setting ()
parser.add_argument('--random_seed', default=123, type=int, help='random_seed')

args = parser.parse_args()

random_seed = args.random_seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)
batch_size = args.batch_size
cutting_layer = args.cutlayer
num_client = args.num_client
save_dir_name = "./{}/{}".format(args.folder, args.filename)

mi = SFL.Trainer(args.arch, cutting_layer, batch_size, n_epochs = args.num_epochs, scheme = args.scheme,
                 num_client = num_client, dataset=args.dataset, save_dir=save_dir_name,
                 regularization_option=args.regularization, regularization_strength = args.regularization_strength,
                 collude_use_public=args.collude_use_public, learning_rate = args.learning_rate,
                 load_from_checkpoint = args.load_from_checkpoint, finetune_freeze_bn = args.finetune_freeze_bn, client_sample_ratio = args.client_sample_ratio,
                 source_task = args.transfer_source_task, load_from_checkpoint_server = args.load_from_checkpoint_server, noniid = args.noniid_ratio)
mi.logger.debug(str(args))

if not os.path.isfile(f"./{args.folder}/{args.filename}/checkpoint_client_best.tar"):
    mi.train(verbose=True)
    mi.resume("./{}/{}/checkpoint_client_best.tar".format(args.folder, args.filename))
else:
    mi.resume("./{}/{}/checkpoint_client_best.tar".format(args.folder, args.filename))

# %%
