import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import models.architectures_torch as architectures
from models.architectures_torch import init_weights
from models.fast_meta import ImagePool, DataIter, reset_l0
from utils import setup_logger, accuracy, AverageMeter, WarmUpLR, TV, l2loss, zeroing_grad
from utils import freeze_model_bn, average_weights
import logging
import torchvision
from datetime import datetime
import os, copy
from shutil import rmtree
from datasets import get_dataset, denormalize, get_image_shape
from models import get_model
from tools import DiffAugment
from kornia import augmentation # differentiable augmentation
import wandb
import tqdm

class Trainer:
    def __init__(self, arch, cutting_layer, batch_size, n_epochs, scheme="V2", num_client=2, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0, learning_rate=0.1, 
                 random_seed=123, load_from_checkpoint = False, attack_confidence_score = False, num_freeze_layer = 0,
                 load_from_checkpoint_server = False, source_task = "cifar100", client_sample_ratio = 1.0,
                 save_activation_tensor = False, noniid = 1.0, last_client_fix_amount = -1, attacker_querying_budget_num_step = -1):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.client_sample_ratio = client_sample_ratio
        self.noniid_ratio = noniid
        self.n_epochs = n_epochs
        # setup save folder
        if save_dir is None:
            self.save_dir = "./saves/{}/".format(datetime.today().strftime('%m%d%H%M'))
        else:
            self.save_dir = str(save_dir) + "/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.save_activation_tensor = save_activation_tensor

        # setup logger
        model_log_file = self.save_dir + '/train.log'
        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger('{}_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
        
        self.warm = 1
        self.scheme = scheme
        
        self.dataset = dataset
        self.image_shape = get_image_shape(self.dataset)
        self.call_resume = False
        self.load_from_checkpoint = load_from_checkpoint
        self.load_from_checkpoint_server = load_from_checkpoint_server
        self.source_task = source_task
        self.cutting_layer = cutting_layer
        self.confidence_score = attack_confidence_score

        # ME attack parameter
        self.last_client_fix_amount = last_client_fix_amount
        self.attacker_querying_budget_num_step = attacker_querying_budget_num_step
        # Activation Defense:
        self.regularization_option = regularization_option

        # If strength is 0.0, then there is no regularization applied, train normally.
        self.regularization_strength = regularization_strength

        # dividing datasets to actual number of clients, self.num_clients is num of active clients at each round (assume client sampling).
        # number of client < actual number of clients

        self.actual_num_users = num_client
        self.num_client = int(num_client * self.client_sample_ratio) # max num of active clients at each round

        print(f"sample {self.num_client} out of {self.actual_num_users}")

        if "gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option: 
            #data-free GAN-attack
            self.actual_num_users = self.actual_num_users - 1 # we let first N-1 client divide the training data, and skip the last client.

        #setup datset
        self.client_dataloader, self.pub_dataloader, self.num_class = get_dataset(self.dataset, self.batch_size, self.noniid_ratio, self.actual_num_users, False, last_client_fix_amount)



        if "gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option: 
            #data-free GAN-attack
            self.actual_num_users = self.actual_num_users + 1 # we let first N-1 client divide the training data, and skip the last client.

        self.attacker_client_id = self.actual_num_users - 1

        self.num_batches = len(self.client_dataloader[0])
        print("Total number of batches per epoch for each client exccept attacker client is ", self.num_batches)

        if "soft_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option: 
            print(f"Total number of batches per epoch for the attacking client (client-{self.attacker_client_id}) is ", len(self.client_dataloader[self.attacker_client_id]))
        
        if self.attacker_querying_budget_num_step != -1: # if this is set (then querying budget is not self.num_batches)
            print(f"Querying budget for the attacking client (client-{self.attacker_client_id}) is ", self.attacker_querying_budget_num_step)

        self.model = get_model(self.arch, self.cutting_layer, self.num_client, self.num_class, num_freeze_layer)
        self.model.merge_classifier_cloud()
        self.model.cloud.cuda()
        self.params = list(self.model.cloud.parameters())
        
        self.local_params = []
        if cutting_layer > 0:
            for i in range(self.num_client):
                self.model.local_list[i].cuda()
                self.local_params.append(self.model.local_list[i].parameters())

        if "surrogate" in self.regularization_option:
            train_clas_layer = self.model.get_num_of_cloud_layer()

            self.surrogate_model = architectures.create_surrogate_model(arch, cutting_layer, self.num_class, self.model.get_num_of_cloud_layer(), "same")

            self.surrogate_model.resplit(train_clas_layer)
            self.surrogate_model.local.apply(init_weights)
            self.surrogate_model.cloud.apply(init_weights)

            # let them be the same model
            self.surrogate_model.local = self.model.local_list[-1]
            self.surrogate_model.cloud.cuda()
            self.surrogate_model.cloud.train()
            self.suro_optimizer = torch.optim.SGD(self.surrogate_model.cloud.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            # milestones = [60, 120, 160]
            milestones = [60, 120]
            self.suro_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.suro_optimizer, milestones=milestones,
                                                        gamma=0.2)  # learning rate decay


        if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
            self.nz = 512
            if "unconditional" not in self.regularization_option:
                self.generator = architectures.GeneratorC(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2])
            else:
                self.generator = architectures.GeneratorD(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2])
            if "multiGAN" in self.regularization_option:
                if "unconditional" not in self.regularization_option:
                    self.generator = architectures.GeneratorC_mult(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2])
                else:
                    self.generator = architectures.GeneratorD_mult(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2])
                print(self.generator)
            if "dynamicGAN_A" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_A(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            if "dynamicGAN_B" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_B(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            if "dynamicGAN_C" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_C(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            
            self.generator.cuda()
            
            if "fast_meta" in self.regularization_option:
                self.data_pool = ImagePool(root=self.save_dir + "/run/")
                CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
                CIFAR10_TRAIN_STD = (0.247, 0.243, 0.261)
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
                ])
                self.aug = transforms.Compose([ 
                    augmentation.RandomCrop(size=[self.image_shape[-2], self.image_shape[-1]], padding=4),
                    augmentation.RandomHorizontalFlip(),
                    transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                ])
                # num of steps to train the surrogate model

                self.kd_step = 5
                
            if "cmi" in self.regularization_option:
                # self.data_pool = ImagePool(root=self.save_dir + "/run/")
                from models.cmi import MemoryBank, MLPHead, MultiTransform
                self.bank_size = 40960
                self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2 * np.prod(self.model.get_smashed_data_size())) # local + global

                self.head = MLPHead(np.prod(self.model.get_smashed_data_size()), 128).cuda().train()
                self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=0.1)
                CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
                CIFAR10_TRAIN_STD = (0.247, 0.243, 0.261)
                self.aug = MultiTransform([
                    # global view
                    transforms.Compose([
                        augmentation.RandomCrop(size=[self.image_shape[-2], self.image_shape[-1]], padding=4),
                        augmentation.RandomHorizontalFlip(),
                        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    ]),
                    # local view
                    transforms.Compose([
                        augmentation.RandomResizedCrop(size=[self.image_shape[-2], self.image_shape[-1]], scale=[0.25, 1.0]),
                        augmentation.RandomHorizontalFlip(),
                        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD),
                    ]),
                ])
                self.n_neg = 4096
                self.cr_T = 0.1
                self.cr = 0.8
                
                # num of steps to train the surrogate model
                self.kd_step = 40

            self.generator_optimizer = torch.optim.Adam(list(self.generator.parameters()), lr=2e-4, betas=(0.5, 0.999))
            # self.generator_optimizer = torch.optim.Adam(list(self.generator.parameters()), lr=5e-5)

            if "EMA" in self.regularization_option:
                self.slow_generator = copy.deepcopy(self.generator)
                for param_t in self.slow_generator.parameters():
                    param_t.requires_grad = False  # not update by gradient
        
        # setup optimizers
        self.optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        
        milestones = [60, 120, 160]
        if self.client_sample_ratio < 1.0:
            for i in range(len(milestones)):
                milestones[i] = int(milestones[i] /self.client_sample_ratio)
        
        self.local_optimizer_list = []
        self.train_local_scheduler_list = []
        self.warmup_local_scheduler_list = []
        for i in range(len(self.local_params)):
            self.local_optimizer_list.append(torch.optim.SGD(list(self.local_params[i]), lr=self.lr, momentum=0.9, weight_decay=5e-4))
            self.train_local_scheduler_list.append(torch.optim.lr_scheduler.MultiStepLR(self.local_optimizer_list[i], milestones=milestones,
                                                                    gamma=0.2))  # learning rate decay
            self.warmup_local_scheduler_list.append(WarmUpLR(self.local_optimizer_list[i], self.num_batches * self.warm))

        self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                                    gamma=0.2)  # learning rate decay
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.num_batches * self.warm)    

    
    def resume(self, model_path_f=None):
        if model_path_f is None:
            model_path_name = self.save_dir + "checkpoint_client_{}.tar".format(self.n_epochs)
        else:
            model_path_name = model_path_f
        
        f = open(self.save_dir + "train.log", "r")
        log_file_txt = f.read(500)
        print("=====Check Original Training LOG=====")
        print(log_file_txt)

        original_cut_layer = int(log_file_txt.split("cutlayer=")[-1].split(",")[0])

        if original_cut_layer != self.cutting_layer:
            print("WARNING: set cutting layer is not the same as in the training setting, proceed with caution.")
            self.model.resplit(original_cut_layer, count_from_right=False)

        for i in range(self.num_client):
            print("load client {}'s local".format(i))
            checkpoint_i = torch.load(model_path_name)
            self.model.local_list[i].cuda()
            self.model.local_list[i].load_state_dict(checkpoint_i, strict = False)

        self.call_resume = True
        print("load cloud")
        checkpoint = torch.load(self.save_dir + "checkpoint_cloud_{}.tar".format(self.n_epochs))
        self.model.cloud.cuda()
        self.model.cloud.load_state_dict(checkpoint, strict = False)
        
        if original_cut_layer != self.cutting_layer:
            self.model.resplit(self.cutting_layer, count_from_right=False)

        self.validate_target()

        # if surrogate model is enabled. load it too
        if "surrogate" in self.regularization_option:
            print("load surrogate client-side")
            checkpoint = torch.load(self.save_dir + "checkpoint_surrogate_client_{}.tar".format(self.n_epochs))
            self.surrogate_model.local.cuda()
            self.surrogate_model.local.load_state_dict(checkpoint, strict = False)
            print("load surrogate server-side")
            checkpoint = torch.load(self.save_dir + "checkpoint_surrogate_cloud_{}.tar".format(self.n_epochs))
            self.surrogate_model.cloud.cuda()
            self.surrogate_model.cloud.load_state_dict(checkpoint, strict = False)

            self.validate_surrogate()

    def sync_client(self, idxs_users = None):
        
        if idxs_users is not None:
            active_local_list = [self.model.local_list[idx] for idx in idxs_users]
        else:
            active_local_list = self.model.local_list

        global_weights = average_weights(active_local_list)
        
        for i in range(self.num_client):
            self.model.local_list[i].load_state_dict(global_weights)

    "Training"
    def train(self, verbose=False): # Train SFL
        
        # start a new wandb run to track this script
        if "train_ME" in self.regularization_option:
            wandb_name = "online"
        else:
            wandb_name = "standard"
        
        
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"{wandb_name}-SFL-Train-fixed-amount",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": self.lr,
            "architecture": self.arch,
            "dataset": self.dataset,
            "last_client_fix_amount": self.last_client_fix_amount,
            "query_budget": self.attacker_querying_budget_num_step,
            "noniid_ratio": self.noniid_ratio,
            "epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "scheme": self.scheme,
            "num_client": self.num_client,
            "cutting_layer": self.cutting_layer,
            "regularization_option": self.regularization_option,
            "regularization_strength": self.regularization_strength,
            }
        )

        '''Train time Model Extraction attack'''
        # we let client collect gradient during SFL training, which is used later for MEA
        # Use a trigger to decide when to start collect gradient

        self.attack_start_epoch = 0
        try:
            self.attack_start_epoch = int(self.regularization_option.split("start")[1])
        except:
            self.logger.debug("extraing start epoch setting from arg, failed, set start_epoch to 0")
            self.attack_start_epoch = 0
        
        self.query_image_id = 0
        self.rotate_label = 0
        if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
            extra_txt = f"gan_train_ME latent vector size is {self.nz}"
        elif "craft_train_ME" in self.regularization_option:
            self.craft_image_id = 0
            self.craft_step_count = 0
            self.num_craft_step = max(int(self.regularization_strength), 20)
            extra_txt = f"Graft_train_ME num_step per images is {self.num_craft_step}"
        elif "GM_train_ME" in self.regularization_option:
            self.GM_data_proportion = self.regularization_strength # use reguarlization_strength to set data_proportion
            extra_txt = f"GM_train_ME data proportion is {self.GM_data_proportion}"
        elif "soft_train_ME" in self.regularization_option:
            extra_txt = f"soft_train_ME data proportion is {1./self.actual_num_users}"
        elif "naive_train_ME" in self.regularization_option:
            extra_txt = "naive_train_ME"
        else:
            extra_txt = "(Not a train_ME run)"

        if self.attacker_querying_budget_num_step != -1:
            num_query =  (self.n_epochs - self.attack_start_epoch) * self.attacker_querying_budget_num_step * self.batch_size
        else:
            num_query =  (self.n_epochs - self.attack_start_epoch) * self.num_batches * self.batch_size
        
        self.logger.debug(f"Perform {self.regularization_option}, total query: {num_query}, starting at {self.attack_start_epoch} epoch, {extra_txt}")

        best_avg_accu = 0.0
        if "surrogate" in self.regularization_option:
            best_avg_surro_accu = 0.0
        if not self.call_resume:
            LOG = np.zeros((self.n_epochs * self.num_batches, self.num_client))

            #load pre-train models
            if self.load_from_checkpoint:
                checkpoint_dir = "./pretrained_models/{}_cutlayer_{}_bottleneck_{}_dataset_{}/".format(self.arch, self.cutting_layer, self.bottleneck_option, self.source_task)
                checkpoint = torch.load(checkpoint_dir + "checkpoint_client_best.tar")
                for i in range(self.num_client):
                    print("load client {}'s local".format(i))
                    self.model.local_list[i].cuda()
                    self.model.local_list[i].load_state_dict(checkpoint)

                print("load cloud")
                checkpoint = torch.load(checkpoint_dir + "checkpoint_cloud_best.tar")
                self.model.cloud.cuda()
                self.model.cloud.load_state_dict(checkpoint)

            self.logger.debug("Train for total {} epochs in {} style".format(self.n_epochs, self.scheme))

            # epoch_save_list = [1, 2 ,5 ,10 ,20 ,50 ,100]
            epoch_save_list = [50, 100, 150, 200]
            if self.client_sample_ratio < 1.0:
                for i in range(len(epoch_save_list)):
                    epoch_save_list[i] = int(epoch_save_list[i] /self.client_sample_ratio)
            #Main Training
            
            
            if "GM_train_ME" in self.regularization_option:
                if self.GM_data_proportion == 0.0:
                    print("TO use GM_train_option, Must have some data available")
                    exit()
                else:
                    if "CIFAR100" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_dataset("cifar100", batch_size=100, actual_num_users=int(1/self.GM_data_proportion))
                    elif "CIFAR10" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_dataset("cifar10", batch_size=100, actual_num_users=int(1/self.GM_data_proportion))
                    elif "SVHN" in self.regularization_option:
                        knockoff_loader_list, _, _ = get_dataset("svhn", batch_size=100, actual_num_users=int(1/self.GM_data_proportion))
                    else: # default use cifar10
                        knockoff_loader_list, _, _ = get_dataset("cifar10", batch_size=100, actual_num_users=int(1/self.GM_data_proportion))

                    knockoff_loader = knockoff_loader_list[0]
                    self.client_dataloader.append(knockoff_loader)

            # save iterator process of actual number of clients
            saved_iterator_list = []
            for client_id in range(len(self.client_dataloader)):
                saved_iterator_list.append(iter(self.client_dataloader[client_id]))
            
            Grad_staleness_visual = False # TODO: set to false.
            if Grad_staleness_visual:
                self.query_image_id = 0

            self.logger.debug("Start SFL training")
            for epoch in range(1, self.n_epochs+1):
                
                
                
                # idxs_users stores a list of client_id. [0, 1, 2, 3, 4]
                if self.client_sample_ratio  == 1.0:
                    idxs_users = range(self.num_client)
                else:
                    idxs_users = np.random.choice(range(self.actual_num_users), self.num_client, replace=False) # 10 out of 1000
                    
                # sample num_client for parapllel training from actual number of users, take their iterator as well.
                client_iterator_list = []
                for client_id in idxs_users:
                    if ("gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option) and client_id == self.attacker_client_id:
                        client_iterator_list.append(None)
                    else:
                        client_iterator_list.append(saved_iterator_list[client_id])

                
                if Grad_staleness_visual:
                    self.rotate_label = -1
                    self.query_image_id += 1
                    images = torch.load("./saved_tensors/test_cifar10_image.pt").cuda()
                    labels = torch.load("./saved_tensors/test_cifar10_label.pt").cuda()
                    self.train_target_step(images, labels, 0, epoch, batch, save_grad = True, skip_regularization=True)
                    self.optimizer_zero_grad()
                
                ## Secondary Loop
                for batch in range(self.num_batches):
                    if self.scheme == "V1":
                        self.optimizer_zero_grad()
                    
                    # if the attacker clients have less data
                    if self.attacker_querying_budget_num_step != -1:
                        # if batch > self.attacker_querying_budget_num_step: # train num_steps on attackers data at the beginning of the epoch
                        if self.num_batches - batch > self.attacker_querying_budget_num_step: # train num_steps on attackers data at the end of the epoch
                            # search for attacker_client_id in current user pool and delete it:
                            if self.attacker_client_id in idxs_users:
                                idxs_users.remove(self.attacker_client_id)
                    
                    
                    for id, client_id in enumerate(idxs_users): # id is the position in client_iterator_list, client_id is the actual client id.
                        # Get data
                        # print(f"id is {id}, client_id is {client_id}")
                        if client_id != self.attacker_client_id: # if current client is not the attack client (default is the last one)
                            try:
                                images, labels = next(client_iterator_list[id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[id] = iter(self.client_dataloader[client_id])
                                    images, labels = next(client_iterator_list[id])
                            except StopIteration:
                                client_iterator_list[id] = iter(self.client_dataloader[client_id])
                                images, labels = next(client_iterator_list[id])

                            
                        else: # if the client is the attacker client:
                            images, labels = self.get_data_MEA_client(client_iterator_list, id, client_id, epoch)
                        
                        if self.scheme == "V2":
                            self.optimizer_zero_grad()
                        
                        
                        if client_id != self.attacker_client_id:
                            # Train step (client/server)
                            train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = False, skip_regularization = False)
                        else: 
                            # MEA client, perform training, collecting gradients using the adv data
                            if epoch > self.attack_start_epoch: # after the attack starts
                                if "craft_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.craft_train_target_step(id, epoch, batch)
                                elif "gan_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.gan_train_target_step(id, self.batch_size, epoch, batch)
                                elif "gan_assist_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.gan_assist_train_target_step(images, labels, id, epoch, batch)
                                elif "GM_train_ME" in self.regularization_option or "soft_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = True, skip_regularization=True) # adv clients won't comply to the defense
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = False, skip_regularization = False)
                            else: # before the attack starts
                                if "gan_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option:
                                    pass # do nothing prior to the starting epoch   
                                elif "soft_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option: # adv clients won't comply to the defense
                                    train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = False, skip_regularization=False)
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = False, skip_regularization = False)

                        if self.scheme == "V2":
                            self.optimizer_step()
                        
                        # Logging
                        if verbose and batch % self.num_batches == 0:
                            self.logger.debug(
                                "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                    epoch, self.n_epochs, batch, self.num_batches, client_id, train_loss, f_loss))
                            wandb.log({f"client{client_id}-train-loss": train_loss})
                        # increment rotate_label
                        if ("soft_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option) and epoch > self.attack_start_epoch and idxs_users[client_id] == self.attacker_client_id:  # SoftTrain, rotate labels
                            self.rotate_label += 1
                    if self.scheme == "V1":
                        self.optimizer_step()
                
                # model synchronization
                self.sync_client()
                    
                
                # Validate and get average accu among clients
                avg_accu = 0
                avg_accu, loss = self.validate_target(client_id=0)
                wandb.log({f"val_acc": avg_accu, "val_loss": loss})
                
                if "surrogate" in self.regularization_option:
                    surro_accu, loss = self.validate_surrogate()
                    wandb.log({f"surrogate_acc": surro_accu, "surrogate_val_loss": loss})
                    if surro_accu > best_avg_surro_accu:
                        best_avg_surro_accu = surro_accu
                
                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    best_avg_accu = avg_accu

                # Save Model regularly
                if epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)
                    if "surrogate" in self.regularization_option:
                        self.save_surrogate(epoch)
                    if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
                        torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))
                # Report Final Accuracy
                if epoch == self.n_epochs:
                    self.logger.debug("Final Target Validation Accuracy is {}".format(avg_accu))
                    wandb.run.summary["Final-Val-Accuracy"] = avg_accu
                    if "surrogate" in self.regularization_option:
                        self.logger.debug("Final Surrogate Validation Accuracy is {}".format(surro_accu))
                        wandb.run.summary["Final-Surrogate-Accuracy"] = surro_accu
        
                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)
                else:
                    self.scheduler_step(epoch)
                    if "surrogate" in self.regularization_option:
                        self.suro_scheduler.step(epoch)
                
        #save final images for gan_train_ME generator.
        if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
            self.generator.cuda()
            self.generator.eval()
            z = torch.randn((10, self.nz)).cuda()
            train_output_path = "{}/generator_final".format(self.save_dir)
            for i in range(self.num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                
                #Get fake image from generator
                if "multiGAN" not in self.regularization_option:
                    fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation
                else:
                    fake_list = []
                    for j in range(len(self.generator.generator_list)):
                        fake_list.append(self.generator.generator_list[j](z, labels))
                    fake = torch.cat(fake_list, dim = 0)

                imgGen = fake.clone()
                imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path):
                    os.mkdir(train_output_path)
                if not os.path.isdir(train_output_path + "/{}".format(self.n_epochs)):
                    os.mkdir(train_output_path + "/{}".format(self.n_epochs))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(self.n_epochs,"final_label{}".format(i)))
            torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))

        if not self.call_resume:
            self.logger.debug("Best Target Validation Accuracy is {}".format(best_avg_accu))
            wandb.run.summary["Best-Val-Accuracy"] = best_avg_accu
            if "surrogate" in self.regularization_option:
                self.logger.debug("Best Surrogate Validation Accuracy is {}".format(best_avg_surro_accu))
                wandb.run.summary["Best-Surrogate-Accuracy"] = best_avg_surro_accu
        else:
            LOG = None
            avg_accu = 0
            avg_accu, _ = self.validate_target(client_id=0)
            self.logger.debug("Best Average Validation Accuracy is {}".format(avg_accu))
            wandb.run.summary["Best-Val-Accuracy"] = avg_accu
        wandb.finish()
        return LOG

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
            self.warmup_scheduler.step(epoch)
            for i in range(len(self.warmup_local_scheduler_list)):
                self.warmup_local_scheduler_list[i].step(epoch)
        else:
            self.train_scheduler.step(epoch)
            for i in range(len(self.train_local_scheduler_list)):
                self.train_local_scheduler_list[i].step(epoch)
    
    def validate_target(self, client_id=0):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        self.model.local_list[client_id].eval()
        self.model.cloud.eval()
        criterion = nn.CrossEntropyLoss()

        if self.arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.Resize(224),
                transforms.CenterCrop(224))

        for i, (input, target) in enumerate(val_loader):
            if self.arch == "ViT":
                input = dl_transforms(input)
            input = input.cuda()
            target = target.cuda()
            # compute output
            with torch.no_grad():
                output = self.model(input)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        self.logger.debug('Test (client-{0}):\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(client_id, loss=losses,top1=top1))
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return top1.avg, losses.avg


    def validate_surrogate(self):
        """
        Run evaluation
        """
        # batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        val_loader = self.pub_dataloader

        # switch to evaluate mode
        self.surrogate_model.local.eval()
        self.surrogate_model.cloud.eval()
        criterion = nn.CrossEntropyLoss()

        if self.arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.Resize(224),
                transforms.CenterCrop(224))

        for i, (input, target) in enumerate(val_loader):
            if self.arch == "ViT":
                input = dl_transforms(input)
            input = input.cuda()
            target = target.cuda()
            # compute output
            with torch.no_grad():
                output = self.surrogate_model(input)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        self.logger.debug('Test (surrogate):\t' 'Loss {loss.val:.4f} ({loss.avg:.4f})\t' 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(loss=losses,top1=top1))
        self.logger.debug(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
        return top1.avg, losses.avg
    
    
    
    
    
    
    def train_target_step(self, x_private, label_private, client_id, epoch, batch, attack = False, skip_regularization = False):

        dl_transforms = torch.nn.Sequential(
            transforms.RandomCrop(self.image_shape[-1], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15))

        if self.arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
        )
        
        if dl_transforms is not None:
            x_private = dl_transforms(x_private)
        
        self.model.cloud.train()
        self.model.local_list[client_id].train()
        
        if attack and "diffaug" in self.regularization_option:
            x_private = torch.cat([DiffAugment.DiffAugment(x_private[:x_private.size(0)//2, :, :, :], 'color,translation,cutout'), x_private[x_private.size(0)//2:, :, :, :]], dim = 0) # a parameterized augmentation module.

        if attack: # if we save grad, meaning we are doing softTrain, which has a poisoning effect, we do not want this to affect aggregation.
            # collect gradient
            if "advnoise" in self.regularization_option:
                if not os.path.isdir(self.save_dir + "/pool_advs"):
                    os.makedirs(self.save_dir + "/pool_advs")
            
            if "surrogate" not in self.regularization_option:
                if not os.path.isdir(self.save_dir + "/saved_grads"):
                    os.makedirs(self.save_dir + "/saved_grads")
                # collect image/label
                if epoch > self.n_epochs - 10 and self.rotate_label == 0:
                    max_allow_image_id = 500 # 500 times batch size is a considerable amount.
                    if self.query_image_id <= max_allow_image_id:
                        torch.save(x_private.detach().cpu(), self.save_dir + f"/saved_grads/image_{self.query_image_id}.pt")
                        torch.save(label_private.detach().cpu(), self.save_dir + f"/saved_grads/label_{self.query_image_id}.pt")

        if attack and "randomnoise" in self.regularization_option: # add a random noise of norm value = regularization_strength to the input.
            
            x_private = x_private + self.regularization_strength * torch.randn_like(x_private)

        
        
        if attack and "advnoise" in self.regularization_option:
            if os.path.isfile(self.save_dir + f"/pool_advs/lastest_batch.pt"):
                adv_sample = torch.load(self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                adv_sample_label = torch.load(self.save_dir + f"/pool_advs/label_lastest_batch.pt")
                x_private = torch.cat([adv_sample[x_private.size(0)//2:, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0) # a parameterized augmentation module.
                label_private = torch.cat([adv_sample_label[x_private.size(0)//2:], label_private[x_private.size(0)//2:, :, :, :]], dim = 0) # a parameterized augmentation module.
        
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        if attack: # if we save grad, meaning we are doing softTrain, which has a poisoning effect, we do not want this to affect aggregation.
            if "advnoise" in self.regularization_option:
                # x_private = torch.Tensor(x_private)
                x_private.requires_grad_(True)

        if self.arch != "ViT":
            # Final Prediction Logits (complete forward pass)
            z_private = self.model.local_list[client_id](x_private)
            if "reduce_grad_freq" in self.regularization_option and batch % 2 == 1:
                z_private = z_private.detach()
            else:
                
                z_private.retain_grad()
            output = self.model.cloud(z_private)
        else:
            output = self.model(x_private)


        if "label_smooth" in self.regularization_option:
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.regularization_strength)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if not skip_regularization: # this is execute by benign clients
            if "l1" in self.regularization_option:
                all_params = torch.cat([x.view(-1) for x in self.model.local_list[client_id].parameters()])
                l1_regularization = self.regularization_strength * torch.norm(all_params, 1)
                total_loss = total_loss + l1_regularization
            if "l2" in self.regularization_option:
                all_params = torch.cat([x.view(-1) for x in self.model.local_list[client_id].parameters()])
                l2_regularization = self.regularization_strength * torch.norm(all_params, 2)
                total_loss = total_loss + l2_regularization
            
            if "gradient_noise" in self.regularization_option and self.arch != "ViT":
                h = z_private.register_hook(lambda grad: grad + 1e-2 * torch.max(torch.abs(grad)) * torch.rand_like(grad).cuda())
        
        total_loss.backward()

        if not skip_regularization:
            if "gradient_noise" in self.regularization_option and self.arch != "ViT":
                h.remove()

        if attack: # if we save grad, meaning we are doing softTrain, which has a poisoning effect, we do not want this to affect aggregation.
            
            if "naive_train_ME" not in self.regularization_option:
                zeroing_grad(self.model.local_list[client_id])
                if "surrogate" not in self.regularization_option:
                    torch.save(z_private.grad.detach().cpu(), self.save_dir + f"/saved_grads/grad_image{self.query_image_id}_label{self.rotate_label}.pt")
            
            if "GM_train_ME" in self.regularization_option:
                if "surrogate" not in self.regularization_option:
                    torch.save(z_private.detach().cpu(), self.save_dir + f"/saved_grads/act_{self.query_image_id}_label{self.rotate_label}.pt")
            
            if "advnoise" in self.regularization_option:
                # gradient = torch.autograd.grad(outputs=f_loss, inputs=x_private, grad_outputs=torch.ones_like(f_loss), retain_graph=True)
                # if x_private.grad is not None:
                #     print(x_private.grad)
                # torch.save(x_private.detach().cpu(), self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                if "signflip" in self.regularization_option and "pgd" in self.regularization_option:
                    torch.save( torch.clip(x_private.detach().cpu() + self.regularization_strength * torch.sign(x_private.grad.detach().cpu()), -1, 1) , self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                elif "signflip" in self.regularization_option:
                    torch.save( torch.clip(x_private.detach().cpu() + self.regularization_strength * x_private.grad.detach().cpu(), -1, 1) , self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                elif "pgd" in self.regularization_option:
                    torch.save( torch.clip(x_private.detach().cpu() - self.regularization_strength * torch.sign(x_private.grad.detach().cpu()), -1, 1) , self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                else:
                    torch.save( torch.clip(x_private.detach().cpu() - self.regularization_strength * x_private.grad.detach().cpu(), -1, 1) , self.save_dir + f"/pool_advs/img_lastest_batch.pt")
                torch.save(label_private.detach().cpu(), self.save_dir + f"/pool_advs/label_lastest_batch.pt")
            
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss
        
        if attack:
            if "surrogate" in self.regularization_option and "naive_train_ME" in self.regularization_option:
                # print(f"train surrogate model by client {client_id}")
                self.surrogate_model.local.eval()
                self.surrogate_model.cloud.train()
                with torch.no_grad():
                    suro_act = self.surrogate_model.local(x_private.detach())
                suro_output = self.surrogate_model.cloud(suro_act)
                suro_loss = criterion(suro_output, label_private)
                self.suro_optimizer.zero_grad()
                suro_loss.backward()
                self.suro_optimizer.step()
                del suro_loss
            elif "surrogate" in self.regularization_option and "GM_train_ME" in self.regularization_option:
                self.surrogate_model.local.eval()
                self.surrogate_model.cloud.train()
                with torch.no_grad():
                    suro_act = self.surrogate_model.local(x_private.detach())
                suro_act.requires_grad = True
                suro_output = self.surrogate_model.cloud(suro_act)
                suro_loss = criterion(suro_output, label_private)

                gradient_loss_style = "l2"
                grad_lambda = 1.0
                grad_approx = torch.autograd.grad(suro_loss, suro_act, create_graph = True)[0]
                if gradient_loss_style == "l2":
                    grad_loss = ((z_private.grad.detach() - grad_approx).norm(dim=1, p =2)).mean()
                elif gradient_loss_style == "l1":
                    grad_loss = ((z_private.grad.detach() - grad_approx).norm(dim=1, p =1)).mean()
                elif gradient_loss_style == "cosine":
                    grad_loss = torch.mean(1 - F.cosine_similarity(grad_approx, z_private.grad.detach(), dim=1))
                suro_grad_loss = grad_loss * grad_lambda
                self.suro_optimizer.zero_grad()

                suro_grad_loss.backward()

                self.suro_optimizer.step()
        return total_losses, f_losses
    
    def craft_train_target_step(self, client_id, epoch, batch):
        lambda_TV = 0.0
        lambda_l2 = 0.0
        craft_LR = 1e-1

        if epoch > 60:
            craft_LR /= 5

        if epoch > 120:
            craft_LR /= 5

        if epoch > 160:
            craft_LR /= 5

        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.cloud.train()
        self.model.local_list[client_id].train()
        image_save_path = self.save_dir + '/craft_pairs/'

        if not os.path.isdir(image_save_path):
            os.makedirs(image_save_path)

        if self.craft_step_count == 0 and self.craft_image_id == 0:
            image_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
            label_shape = [self.batch_size, ]
            fake_image = torch.rand(image_shape, requires_grad=True, device="cuda")
            fake_label = torch.randint(low=0, high = self.num_class, size = label_shape, device="cuda")
        
        elif self.craft_step_count == self.num_craft_step:
            # save latest crafted image
            fake_image = torch.load(image_save_path + f'image_{self.craft_image_id}.pt')
            imgGen = fake_image.clone()
            imgGen = denormalize(imgGen, self.dataset)
            torchvision.utils.save_image(imgGen, image_save_path + '/visual_{}.jpg'.format(self.craft_image_id))

            # reset counters
            self.craft_step_count = 0
            self.craft_image_id += 1
            image_shape = [self.batch_size, self.image_shape[0], self.image_shape[1], self.image_shape[2]]
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

        featureLoss = criterion(output, fake_label)

        TVLoss = TV(fake_image)
        normLoss = l2loss(fake_image)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss

        totalLoss.backward()

        zeroing_grad(self.model.local_list[client_id])

        craft_optimizer.step()
        if "surrogate" not in self.regularization_option:
            torch.save(fake_image.detach().cpu(), image_save_path + f'image_{self.craft_image_id}.pt')
            torch.save(fake_label.cpu(), image_save_path + f'label_{self.craft_image_id}.pt')
            torch.save(z_private.grad.detach().cpu(), image_save_path + f'grad_{self.craft_image_id}.pt')

        self.craft_step_count += 1

        if "surrogate" in self.regularization_option:
            self.surrogate_model.local = self.model.local_list[client_id]
            self.surrogate_model.local.eval()
            self.surrogate_model.cloud.train()
            
            with torch.no_grad():
                suro_act = self.surrogate_model.local(fake_image.detach())
            suro_output = self.surrogate_model.cloud(suro_act)
            # suro_output = self.surrogate_model(fake_image.detach())
            suro_loss = criterion(suro_output, fake_label)
            self.suro_optimizer.zero_grad()
            suro_loss.backward()
            self.suro_optimizer.step()
            del suro_loss

        return totalLoss.detach().cpu().numpy(), featureLoss.detach().cpu().numpy()

    def gan_train_target_step(self, client_id, batch_size, epoch, batch):

        #if enable poison option
        poison_option = False

        if "poison" in self.regularization_option:
            poison_option = True


        self.model.cloud.train()
        self.model.local_list[client_id].train()
        self.generator.cuda()
        self.generator.train()
        
        self.generator_optimizer.zero_grad()
        train_output_path = self.save_dir + "generator_train"
        if os.path.isdir(train_output_path):
            rmtree(train_output_path)
        os.makedirs(train_output_path)


        #Sample Random Noise
        z = torch.randn((batch_size, self.nz)).cuda()
        
        B = batch_size// 2

        labels_l = torch.randint(low=0, high=self.num_class, size = [B, ]).cuda()
        labels_r = copy.deepcopy(labels_l).cuda()
        label_private = torch.stack([labels_l, labels_r]).view(-1)
        
        #Get fake image from generator
        x_private = self.generator(z, label_private) # pre_x returns the output of G before applying the activation

        if "EMA" in self.regularization_option: # if enable EMA, then alternating G and slow G
            self.slow_generator.cuda()
            self.slow_generator.eval()
            x_private_slow = self.slow_generator(z, label_private).detach()

            x_private = torch.cat([x_private[:x_private.size(0)//2, :, :, :], x_private_slow[x_private.size(0)//2:, :, :, :]], dim = 0)

        if poison_option and batch % 2 == 1: #poison step
            label_private = torch.randint(low=0, high=self.num_class, size = [self.batch_size, ]).cuda()
            x_private = x_private.detach()
        
        # record generator_output during training
        if epoch % 50 == 0 and batch == 0:
            imgGen = x_private.clone()
            imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))
        
        
        
        
        z_private = self.model.local_list[client_id](x_private)

        if "reduce_grad_freq" in self.regularization_option and batch % 2 == 1: # server will skip sending back gradients, once per two steps
            z_private = z_private.detach()
        
        output = self.model.cloud(z_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if "gradient_noise" in self.regularization_option:
            h = z_private.register_hook(lambda grad: grad + 1e-2 * torch.max(torch.abs(grad)) * torch.rand_like(grad).cuda())

        if "adversary" in self.regularization_option:
            total_loss.backward(retain_graph = True)
        else:
            total_loss.backward()
        
        self.generator_optimizer.step()
        
        # zero out local model gradients to avoid unecessary poisoning effect
        zeroing_grad(self.model.local_list[client_id])

        if "gradient_noise" in self.regularization_option:
            h.remove()

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        if "surrogate" in self.regularization_option:
            self.surrogate_model.local = self.model.local_list[client_id]
            self.surrogate_model.local.eval()
            self.surrogate_model.cloud.train()
            with torch.no_grad():
                suro_act = self.surrogate_model.local(x_private.detach())
            suro_output = self.surrogate_model.cloud(suro_act)
            suro_loss = criterion(suro_output, label_private)
            
            self.suro_optimizer.zero_grad()
            suro_loss.backward()
            self.suro_optimizer.step()

            if not ("reduce_grad_freq" in self.regularization_option and batch % 2 == 1):
                if "adversary" in self.regularization_option:
                    self.generator_optimizer.zero_grad()

                    # generator get reverse
                    with torch.no_grad():
                        suro_z_private = self.surrogate_model.local(x_private)
                    
                    suro_output = self.surrogate_model.cloud(suro_z_private)

                    suro_loss = criterion(suro_output, label_private)

                    adversary_loss = - 2.0 * suro_loss # maximize this loss.

                    adversary_loss.backward() # this only affect generator
                    
                    self.generator_optimizer.step()

                    del adversary_loss
                

            del suro_loss

        if "EMA" in self.regularization_option:
            with torch.no_grad():
                for online, target in zip(self.generator.parameters(), self.slow_generator.parameters()):
                    target.data = 0.95 * target.data + 0.05 * online.data


        return total_losses, f_losses

    def gan_assist_train_target_step(self, x_private, label_private, client_id, epoch, batch):
        
        dl_transforms = torch.nn.Sequential(
            transforms.RandomCrop(self.image_shape[-1], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15))

        if self.arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
        )
        
        if dl_transforms is not None:
            x_private = dl_transforms(x_private)
        
        # apply diffaug during the training to allow more variation to the input images.
        if "diffaug" in self.regularization_option:
            x_private = DiffAugment.DiffAugment(x_private, 'color,translation,cutout') # a parameterized augmentation module.
        
        #if enable poison option
        self.model.cloud.train()
        self.model.local_list[client_id].train()
        self.generator.cuda()
        self.generator.train()
        
        x_private = x_private.cuda()
        label_private = label_private.cuda()

        train_output_path = self.save_dir + "generator_train"
        if not os.path.isdir(train_output_path):
            os.makedirs(train_output_path)
        
        if "surrogate" not in self.regularization_option:
            if epoch > self.n_epochs - 10 and self.rotate_label == 0:
                max_allow_image_id = 500 # 500 times batch size is a considerable amount.
                if self.query_image_id <= max_allow_image_id:
                    if not os.path.isdir(self.save_dir + "/saved_grads"):
                        os.makedirs(self.save_dir + "/saved_grads")
                    torch.save(x_private.detach().cpu(), self.save_dir + f"/saved_grads/image_{self.query_image_id}.pt")
                    torch.save(label_private.detach().cpu(), self.save_dir + f"/saved_grads/label_{self.query_image_id}.pt")

        

        # Use fast meta
        if "fast_meta" in self.regularization_option:
            self.z = torch.randn(size=(self.batch_size, self.nz), device="cuda:0").requires_grad_() 
            self.generator_optimizer = torch.optim.Adam([
                {'params': self.generator.parameters()},
                {'params': [self.z], 'lr': 0.015}
            ], lr=5e-3, betas=[0.5, 0.999])
            if epoch == 120 and batch == 0:
                reset_l0(self.generator)
            
            if epoch % 5 == 0:
                self.data_pool.reset()
            x_noise = self.generator(self.z[:x_private.size(0)//2, :], label_private[:x_private.size(0)//2]) # pre_x returns the output of G before applying the activation
            
        
        # Use CMI
        elif "cmi" in self.regularization_option:
            self.generator_optimizer = torch.optim.Adam([
                {'params': self.generator.parameters()}
            ], lr=5e-3, betas=[0.5, 0.999])
            if epoch == 120 and batch == 0:
                reset_l0(self.generator)
            z = torch.randn((x_private.size(0)//2, self.nz)).cuda()
            x_noise = self.generator(z, label_private[:x_private.size(0)//2]) # pre_x returns the output of G before applying the activation
        else:
            #Get class-dependent noise, adding to x_private lately
            z = torch.randn((x_private.size(0)//2, self.nz)).cuda()
            x_noise = self.generator(z, label_private[:x_private.size(0)//2]) # pre_x returns the output of G before applying the activation


        # Use EMA
        if "EMA" in self.regularization_option: # if enable EMA, then alternating G and slow G
            self.slow_generator.cuda()
            self.slow_generator.eval()
            x_noise_slow = self.slow_generator(z, label_private[:x_private.size(0)//2]).detach()
            x_noise = torch.cat([x_noise[:x_noise.size(0)//2, :, :, :], x_noise_slow[x_noise.size(0)//2:, :, :, :]], dim = 0)

        # Mixup noise and training images
        x_fake = torch.cat([self.regularization_strength * x_noise + (1 - self.regularization_strength) * x_private[:x_private.size(0)//2, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
        
        # apply diffaug after the mixup to allow more variation to the input images.
        if "diffallaug" in self.regularization_option:
            x_fake = DiffAugment.DiffAugment(x_fake, 'color,translation,cutout') # a parameterized augmentation module.


        #augmentation:
        if "fast_meta" in self.regularization_option:
            x_fake = self.aug(x_fake)
        elif "cmi" in self.regularization_option:
            x_fake, x_fake_local = self.aug(x_fake)
        
        # record generator_output during training
        if epoch % 50 == 0 and batch == 0:
            imgGen = x_noise.clone()
            imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))
        

        z_private = self.model.local_list[client_id](x_fake)
        if "reduce_grad_freq" in self.regularization_option and batch % 2 == 1:
            z_private = z_private.detach()
        elif "GM" in self.regularization_option:
            z_private.retain_grad()
        output = self.model.cloud(z_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if "cmi" in self.regularization_option:
            if x_private.size(0) != self.batch_size:
                pass
            else:
                global_feature = z_private.detach().view(x_private.size(0), -1)
                with torch.no_grad():
                    local_feature = self.model.local_list[client_id](x_fake_local).view(x_private.size(0), -1)
                cached_feature, _ = self.mem_bank.get_data(self.n_neg)
                cached_local_feature, cached_global_feature = torch.chunk(cached_feature.cuda(), chunks=2, dim=1)
                proj_feature = self.head( torch.cat([local_feature, cached_local_feature, global_feature, cached_global_feature], dim=0) )
                proj_local_feature, proj_global_feature = torch.chunk(proj_feature, chunks=2, dim=0)
                # A naive implementation of contrastive loss
                cr_logits = torch.mm(proj_local_feature, proj_global_feature.detach().T) / self.cr_T # (N + N') x (N + N')
                cr_labels = torch.arange(start=0, end=len(cr_logits), device="cuda:0")
                loss_cr = F.cross_entropy( cr_logits, cr_labels, reduction='none')  #(N + N')
                if self.mem_bank.n_updates>0:
                    loss_cr = loss_cr[:x_private.size(0)].mean() + loss_cr[x_private.size(0):].mean()
                else:
                    loss_cr = loss_cr.mean()

                total_loss += self.cr * loss_cr
                self.mem_bank.add( torch.cat([local_feature.data, global_feature.data], dim=1).data )

        if "reg" in self.regularization_option:
            lambda_TV = 6e-3
            lambda_l2 = 1.5e-5
            total_loss += lambda_TV * TV(x_fake)
            total_loss += lambda_l2 * l2loss(x_fake)
        
        if "gradient_noise" in self.regularization_option:
            # see https://medium.com/analytics-vidhya/pytorch-hooks-5909c7636fb
            h = z_private.register_hook(lambda grad: grad + 1e-2 * torch.max(torch.abs(grad)) * torch.rand_like(grad).cuda())

        self.generator_optimizer.zero_grad()
        if "cmi" in self.regularization_option:
            self.optimizer_head.zero_grad()
        
        if "adversary" in self.regularization_option or "GM" in self.regularization_option:
            total_loss.backward(retain_graph=True)
        else:
            total_loss.backward()
        self.generator_optimizer.step()
        
        
        if "cmi" in self.regularization_option:
            self.optimizer_head.step()


        if "gradient_noise" in self.regularization_option:
            h.remove()
        
        
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        if "surrogate" in self.regularization_option:
            self.surrogate_model.local = self.model.local_list[client_id]
            self.surrogate_model.local.eval()
            self.surrogate_model.cloud.train()


            if "fast_meta" in self.regularization_option:
                self.data_pool.add(x_fake.data.cpu(), label_private.data.cpu())
                dst = self.data_pool.get_dataset(transform=self.transform)
                loader = torch.utils.data.DataLoader(
                    dst, batch_size=self.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, sampler=None)
                self.data_iter = DataIter(loader)

                for _ in range(self.kd_step):
                    syns_image, syns_label = self.data_iter.next()
                    syns_image = syns_image.cuda()
                    syns_label = syns_label.cuda()
                    with torch.no_grad():
                        suro_act = self.surrogate_model.local(syns_image.detach())
                    suro_output = self.surrogate_model.cloud(suro_act)
                    suro_loss = criterion(suro_output, syns_label)
                    self.suro_optimizer.zero_grad()
                    suro_loss.backward()
                    self.suro_optimizer.step()

            else:
            
                with torch.no_grad():
                    suro_act = self.surrogate_model.local(x_fake.detach())
                suro_output = self.surrogate_model.cloud(suro_act)
                suro_loss = criterion(suro_output, label_private)

                self.suro_optimizer.zero_grad()
                suro_loss.backward()
                self.suro_optimizer.step()

            if "GM" in self.regularization_option:
                target_grad = z_private.grad.detach()

                with torch.no_grad():
                    suro_act = self.surrogate_model.local(x_fake.detach())
                suro_act.requires_grad = True
                
                
                suro_output = self.surrogate_model.cloud(suro_act)
                suro_loss = criterion(suro_output, label_private)

                grad_approx = torch.autograd.grad(suro_loss, suro_act, create_graph = True)[0]

                grad_loss = ((target_grad - grad_approx).norm(dim=1, p =2)).mean()

                self.suro_optimizer.zero_grad()

                grad_loss.backward()

                self.suro_optimizer.step()

            if not ("reduce_grad_freq" in self.regularization_option and batch % 2 == 1):
                if "adversary" in self.regularization_option:
                    
                    # adversarial generate hard samples
                    self.generator_optimizer.zero_grad()
                    
                    # generator get reverse
                    with torch.no_grad():
                        suro_z_private = self.surrogate_model.local(x_fake)
                    suro_output = self.surrogate_model.cloud(suro_z_private)

                    suro_loss = criterion(suro_output, label_private)

                    adversary_loss = - 2.0 * suro_loss # maximize this loss.
                    
                    if "GM" in self.regularization_option:
                        adversary_loss.backward(retain_graph=True) # this only affect generator
                    else:
                        adversary_loss.backward()
                    
                    self.generator_optimizer.step()

                    # print(f"adversarial loss is {adversary_loss.item()}")

                    del adversary_loss
                
                    if "GM" in self.regularization_option:
                        target_grad = z_private.grad.detach()

                        # adversarial gradient matching
                        self.generator_optimizer.zero_grad()
                        
                        # generator get reverse
                        with torch.no_grad():
                            suro_z_private = self.surrogate_model.local(x_fake)
                        suro_z_private.requires_grad = True
                        suro_output = self.surrogate_model.cloud(suro_z_private)

                        suro_loss = criterion(suro_output, label_private)

                        gradient_loss_style = "l2"
                        grad_approx = torch.autograd.grad(suro_loss, suro_z_private, create_graph = True)[0]
                        if gradient_loss_style == "l2":
                            grad_loss = - ((target_grad - grad_approx).norm(dim=1, p =2)).mean()
                        elif gradient_loss_style == "l1":
                            grad_loss = - ((target_grad - grad_approx).norm(dim=1, p =1)).mean()
                        elif gradient_loss_style == "cosine":
                            grad_loss = - torch.mean(1 - F.cosine_similarity(grad_approx, target_grad, dim=1))
                        
                        grad_loss.backward() # this only affect generator
                        
                        self.generator_optimizer.step()

                        del grad_loss
                
            del suro_loss
        
        

        if "EMA" in self.regularization_option:
            with torch.no_grad():
                for online, target in zip(self.generator.parameters(), self.slow_generator.parameters()):
                    target.data = 0.95 * target.data + 0.05 * online.data
        
        return total_losses, f_losses

    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"
        torch.save(self.model.local_list[0].state_dict(), self.save_dir + 'checkpoint_client_{}.tar'.format(epoch))
        torch.save(self.model.cloud.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))

    def save_surrogate(self, epoch, is_best=False):
        if is_best:
            epoch = "best"
        torch.save(self.surrogate_model.local.state_dict(), self.save_dir + 'checkpoint_surrogate_client_{}.tar'.format(epoch))
        torch.save(self.surrogate_model.cloud.state_dict(), self.save_dir + 'checkpoint_surrogate_cloud_{}.tar'.format(epoch))

    def get_data_MEA_client(self, client_iterator_list, id, client_id, epoch):

        self.old_image = None
        self.old_label = None

        if ("gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option):   # Data free no data.
            
            return None, None
        
        elif ("soft_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option) and epoch > self.attack_start_epoch:  # SoftTrain, attack mode
            
            if self.rotate_label == self.num_class or (self.query_image_id == 0 and self.rotate_label == 0):
                
                if (self.query_image_id == 0 and self.rotate_label == 0):
                    client_iterator_list[id] = iter(self.client_dataloader[client_id])
                
                if self.rotate_label == self.num_class:
                    self.rotate_label = 0
                    self.query_image_id += 1
                
                try:
                    self.old_images, self.old_labels = next(client_iterator_list[id])
                except StopIteration:
                    client_iterator_list[id] = iter(self.client_dataloader[client_id])
                    self.old_images, self.old_labels = next(client_iterator_list[id])
                
                if "GM_train_ME" in self.regularization_option:
                    # self.old_labels = (torch.randint_like(self.old_labels, low = 0, high = 10) + int(self.rotate_label)) % self.num_class
                    # self.old_labels = (torch.randint_like(self.old_labels, low = 0, high = 10) + int(self.rotate_label)) % self.num_class

                    self.old_labels = torch.ones_like(self.old_labels) * int(self.rotate_label)
                elif "soft_train_ME" in self.regularization_option:
                    self.old_labels = (self.old_labels + int(self.rotate_label)) % self.num_class # rotating around orginal true label
            
            else: # rotate labels
                self.old_labels = (self.old_labels + 1) % self.num_class # add 1 to label
            
            return self.old_images, self.old_labels

        else: # before attack_start_epoch, submit benigh images if possible
            
            if "GM_train_ME" in self.regularization_option:
                return None, None

            try:
                images, labels = next(client_iterator_list[id])
                if images.size(0) != self.batch_size:
                    client_iterator_list[id] = iter(self.client_dataloader[client_id])
                    images, labels = next(client_iterator_list[id])
            except StopIteration:
                client_iterator_list[id] = iter(self.client_dataloader[client_id])
                images, labels = next(client_iterator_list[id])

            if epoch > self.n_epochs - 10:
                if ("gan_assist_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option):
                    self.query_image_id += 1
            
            
            return images, labels