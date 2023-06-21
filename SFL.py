import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import models.architectures_torch as architectures
from models.architectures_torch import init_weights
from utils import setup_logger, accuracy, AverageMeter, WarmUpLR, TV, l2loss, zeroing_grad, fidelity
from utils import average_weights
import logging
import torchvision
from datetime import datetime
import os, copy
from shutil import rmtree
from datasets import get_dataset, denormalize, get_image_shape
from models import get_model
import wandb
import tqdm
import gc

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
        else:
            print(f"Querying budget for the attacking client (client-{self.attacker_client_id}) is not set, but gets capped at {self.num_batches}")

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
                num_generator = 10
                self.logger.debug(f"Use multiGAN with number of generator {num_generator}")
                if "unconditional" not in self.regularization_option:
                    self.generator = architectures.GeneratorC_mult(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_generator = num_generator)
                else:
                    self.generator = architectures.GeneratorD_mult(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_generator = num_generator)
            
            if "multiresGAN" in self.regularization_option:
                num_generator = 10
                self.generator = architectures.Generator_resC_mult(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_generator = num_generator)
            
            if "dynamicGAN_A" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_A(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            if "dynamicGAN_B" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_B(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            if "dynamicGAN_C" in self.regularization_option:
                self.generator = architectures.GeneratorDynamic_C(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2], num_heads = 50)
            
            self.generator.cuda()

            self.generator_optimizer = torch.optim.Adam(list(self.generator.parameters()), lr=2e-4, betas=(0.5, 0.999))
        
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
            print(f"Resplit to {original_cut_layer} layers in client-side model")
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

            self.fidelity_test()
        
        if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
            
            print("load generator")
            checkpoint = torch.load(self.save_dir + "checkpoint_generator_{}.tar".format(self.n_epochs))
            self.generator.cuda()
            self.generator.load_state_dict(checkpoint, strict = False)

            mean_var = self.generator_eval()
        
            self.logger.debug("Final-Generator_VAR: {}".format(mean_var))

            mean_margin = self.generator_margin_eval()
        
            self.logger.debug("Final-Generator_MEAN_MARGIN: {}".format(mean_margin))

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
            epoch_save_list = [200]
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
                    idxs_users = list(range(self.num_client))
                else:
                    idxs_users = np.random.choice(range(self.actual_num_users), self.num_client, replace=False).tolist() # 10 out of 1000
                    
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
                    
                    
                    
                    
                    for id, client_id in enumerate(idxs_users): # id is the position in client_iterator_list, client_id is the actual client id.
                        
                        # for gan_train_ME, we skip the attacker_client if we set the attacker_querying_budget_num_step
                        if "gan_train_ME" in self.regularization_option:
                            if self.attacker_querying_budget_num_step != -1:
                                # if batch > self.attacker_querying_budget_num_step: # train num_steps on attackers data at the beginning of the epoch
                                if self.num_batches - batch > self.attacker_querying_budget_num_step: # train num_steps on attackers data at the end of the epoch
                                    # skip attacking client in this step
                                    if client_id == self.attacker_client_id:
                                        continue
                        
                        # Get data
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

                                    # for gan_train_ME, we change to normal training if we set the attacker_querying_budget_num_step, to keep the other variants the same
                                    if self.attacker_querying_budget_num_step != -1 and self.num_batches - batch > self.attacker_querying_budget_num_step:
                                        train_loss, f_loss = self.train_target_step(images, labels, id, epoch, batch, attack = True, skip_regularization=True)
                                        # if batch > self.attacker_querying_budget_num_step: # train num_steps on attackers data at the beginning of the epoch
                                    else:
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
                        if verbose and batch == self.num_batches -1:
                            self.logger.debug(
                                "log--[{}/{}][{}/{}][client-{}] train loss: {:1.4f} cross-entropy loss: {:1.4f}".format(
                                    epoch, self.n_epochs, batch + 1, self.num_batches, client_id, train_loss, f_loss))
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
                
                gc.collect()
        if "gan" in self.regularization_option:
            #evaluate final images for gan_train_ME generator.
            mean_var = self.generator_eval()
            self.logger.debug("Final-Generator_VAR: {}".format(mean_var))
            wandb.run.summary["Final-Generator_VAR"] = mean_var

            mean_margin = self.generator_margin_eval()
            self.logger.debug("Final-Generator_MEAN_MARGIN: {}".format(mean_margin))
            wandb.run.summary["Final-Generator_MEAN_MARGIN"] = mean_margin

        #evaluate final MEA performance (fidelity test).
        if "surrogate" in self.regularization_option:
            fidel_score = self.fidelity_test()
            self.logger.debug("Final Surrogate Fidelity Score is {}".format(fidel_score))
            wandb.run.summary["Final-Surrogate-Fidelity"] = fidel_score
        
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
                x_private = torch.cat([adv_sample[x_private.size(0)//2:, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0) 
                label_private = torch.cat([adv_sample_label[x_private.size(0)//2:], label_private[x_private.size(0)//2:, :, :, :]], dim = 0)
        
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
            if "nopoison" in self.regularization_option:
                zeroing_grad(self.model.local_list[client_id])
            
            if "naive_train_ME" not in self.regularization_option:
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
        
        if attack:
            
            if "print_grad_stats" in self.regularization_option and batch == 0:
                if not os.path.isdir(self.save_dir + "/grad_stats"):
                    os.makedirs(self.save_dir + "/grad_stats")
                x_private = torch.load(f"./saved_tensors/test_{self.dataset}_image.pt").cuda()
                label_private = torch.load(f"./saved_tensors/test_{self.dataset}_label.pt").cuda()
                # x_private.requires_grad_(True)
                z_private = self.model.local_list[client_id](x_private)
                z_private.retain_grad()
                output = self.model.cloud(z_private)
                f_loss = criterion(output, label_private)
                f_loss.backward()
                torch.save(z_private.grad.detach().cpu(), self.save_dir + f"/grad_stats/grad_epoch_{epoch}_batch0.pt")

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
            
            if "print_margin_stats"  in self.regularization_option and batch == 0:
                if not os.path.isdir(self.save_dir + "/margin_stats"):
                        os.makedirs(self.save_dir + "/margin_stats")
                if "surrogate" in self.regularization_option:
                    confidence_score_margin_surrogate = self.calculate_margin(x_private.detach(), using_surrogate_if_available=True)/x_private.size(0) # margin to the generator

                    file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate.txt", "a")
                    file1.write(f"{confidence_score_margin_surrogate}, ")
                    file1.close()

                confidence_score_margin_target = self.calculate_margin(x_private.detach(), using_surrogate_if_available=False)/x_private.size(0) # margin to the generator

                file2 = open(f"{self.save_dir}/margin_stats/margin_target.txt", "a")
                file2.write(f"{confidence_score_margin_target}, ")
                file2.close()
        
        
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
        
        if "nopoison" in self.regularization_option:
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

        self.model.cloud.train()
        self.model.local_list[client_id].train()
        self.generator.cuda()
        self.generator.train()
        
        self.generator_optimizer.zero_grad()
        train_output_path = self.save_dir + "generator_train"
        if not os.path.isdir(train_output_path):
            os.makedirs(train_output_path)

        #Sample Random Noise
        z = torch.randn((batch_size, self.nz)).cuda()
        
        if "randommix" in self.regularization_option and "test4" in self.regularization_option: # Mixup with images from the correct class
            B = batch_size// 2
            labels_l = torch.randint(low=0, high=self.num_class, size = [B, ]).cuda()
            labels_r = copy.deepcopy(labels_l).cuda()
            label_private = torch.stack([labels_l, labels_r]).view(-1)
        else:
            label_private = torch.randint(low=0, high=self.num_class, size = [batch_size, ]).cuda()

            if "test7" in self.regularization_option or "test8" in self.regularization_option:
                for t in range(batch_size//2):
                    while label_private[t] == label_private[batch_size//2 + t]:
                        label_private[t] = np.random.randint(low = 0, high = self.num_class)

        x_noise = self.generator(z, label_private) # pre_x returns the output of G before applying the activation
        
        #Get fake image from generator
        if "randommix" in self.regularization_option:
            
            if "test1" in self.regularization_option:
                # Mixup g_out and g_out from random classes, reverse the strength position, send mixture together with normal g_out
                x_private = torch.cat([torch.clip(self.regularization_strength * x_noise[:x_noise.size(0)//2, :, :, :] + x_noise[x_noise.size(0)//2:, :, :, :].detach(), -1, 1), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
            elif "test2" in self.regularization_option:
                # Mixup g_out and random images, send mixture together with normal g_out
                x_private = torch.cat([torch.clip(x_noise[:x_noise.size(0)//2, :, :, :] + self.regularization_strength * torch.randn_like(x_noise[:x_noise.size(0)//2, :, :, :]).cuda(), -1, 1), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
            elif "test3" in self.regularization_option: # default option but removing clipping
                x_private = torch.cat([x_noise[:x_noise.size(0)//2, :, :, :] + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
            elif "test6" in self.regularization_option or "test8" in self.regularization_option:
                x_private = torch.cat([(1 - self.regularization_strength) * x_noise[:x_noise.size(0)//2, :, :, :] + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
            else: # default option
                # Mixup g_out and g_out from random classes, send mixture together with normal g_out
                x_private = torch.cat([torch.clip(x_noise[:x_noise.size(0)//2, :, :, :] + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), -1, 1), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
        else:
            x_private = x_noise
        
        
        # record generator_output during training
        if epoch % 20 == 0 and batch == 0:
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            
            imgGen = x_noise.clone()
            imgGen = denormalize(imgGen, self.dataset)
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/gout_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

            if "randommix" in self.regularization_option:
                imgGen2 = (self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach()).clone()
                imgGen2 = denormalize(imgGen2, self.dataset)
                torchvision.utils.save_image(imgGen2, train_output_path + '/{}/out_randomix_additive_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

            imgGen1 = x_private.clone()
            imgGen1 = denormalize(imgGen1, self.dataset)
            torchvision.utils.save_image(imgGen1, train_output_path + '/{}/out_finalout_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

        z_private = self.model.local_list[client_id](x_private)

        if "reduce_grad_freq" in self.regularization_option and batch % 2 == 1: # server will skip sending back gradients, once per two steps
            z_private = z_private.detach()
        output = self.model.cloud(z_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss

        if "gradient_noise" in self.regularization_option:
            h = z_private.register_hook(lambda grad: grad + 1e-2 * torch.max(torch.abs(grad)) * torch.rand_like(grad).cuda())

        total_loss.backward()
        
        self.generator_optimizer.step()
        
        if "gradient_noise" in self.regularization_option:
            h.remove()

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        if "surrogate" in self.regularization_option:
            self.surrogate_model.local = self.model.local_list[client_id]
            self.surrogate_model.cloud.train()

            if "test5" in self.regularization_option:
                surrogate_input = torch.cat([x_noise[:x_noise.size(0)//2, :, :, :], torch.clip(x_noise[:x_noise.size(0)//2, :, :, :] + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), -1, 1), x_noise[x_noise.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = torch.cat([label_private[:x_private.size(0)//2], label_private], dim = 0)
            else:
                surrogate_input = x_private
                surrogate_label = label_private
            
            with torch.no_grad():
                suro_act = self.surrogate_model.local(surrogate_input.detach())
            suro_output = self.surrogate_model.cloud(suro_act)
            suro_loss = criterion(suro_output, surrogate_label)
            
            self.suro_optimizer.zero_grad()
            suro_loss.backward()
            self.suro_optimizer.step()
        
        # zero out local model gradients to avoid unecessary poisoning effect #It turns out the poisoning effect helps accelerate the attack.
        if "nopoison" in self.regularization_option:
            zeroing_grad(self.model.local_list[client_id])
        
        if "print_margin_stats"  in self.regularization_option and batch == 0:
            if not os.path.isdir(self.save_dir + "/margin_stats"):
                    os.makedirs(self.save_dir + "/margin_stats")
            if "surrogate" in self.regularization_option:

                confidence_score_margin_surrogate = self.calculate_margin(x_noise.detach(), using_surrogate_if_available=True)/x_noise.size(0) # margin to the generator
                file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate_on_noise.txt", "a")
                file1.write(f"{confidence_score_margin_surrogate}, ")
                file1.close()

                if "randommix" in self.regularization_option:
                    confidence_score_margin_surrogate = self.calculate_margin(torch.clip(x_noise[:x_noise.size(0)//2, :, :, :].detach() + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), -1, 1), using_surrogate_if_available=True) / (x_noise.size(0)//2) # margin to the generator
                    file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate_on_mixture.txt", "a")
                    file1.write(f"{confidence_score_margin_surrogate}, ")
                    file1.close()

                confidence_score_margin_surrogate = self.calculate_margin(surrogate_input.detach(), using_surrogate_if_available=True)/surrogate_input.size(0) # margin to the generator
                file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate.txt", "a")
                file1.write(f"{confidence_score_margin_surrogate}, ")
                file1.close()

            confidence_score_margin_target = self.calculate_margin(x_noise.detach(), using_surrogate_if_available=False)/x_noise.size(0) # margin to the generator
            file2 = open(f"{self.save_dir}/margin_stats/margin_target_on_noise.txt", "a")
            file2.write(f"{confidence_score_margin_target}, ")
            file2.close()

            if "randommix" in self.regularization_option:
                confidence_score_margin_target = self.calculate_margin(torch.clip(x_noise[:x_noise.size(0)//2, :, :, :].detach() + self.regularization_strength * x_noise[x_noise.size(0)//2:, :, :, :].detach(), -1, 1), using_surrogate_if_available=False) / (x_noise.size(0)//2) # margin to the generator
                file2 = open(f"{self.save_dir}/margin_stats/margin_target_on_mixture.txt", "a")
                file2.write(f"{confidence_score_margin_target}, ")
                file2.close()

            confidence_score_margin_target = self.calculate_margin(x_private.detach(), using_surrogate_if_available=False)/x_private.size(0) # margin to the generator
            file2 = open(f"{self.save_dir}/margin_stats/margin_target.txt", "a")
            file2.write(f"{confidence_score_margin_target}, ")
            file2.close()




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
        
        #Get class-dependent noise, adding to x_private lately
        z = torch.randn((x_private.size(0)//2, self.nz)).cuda()

        if "test12" in self.regularization_option or "test13" in self.regularization_option:
            #change the second half of label_private, to avoid any overlap with the first half (the random images we want it to mix with):
            
            noise_label = label_private[x_private.size(0)//2:].clone()
            for t in range(x_private.size(0)//2):
                while label_private[t] == noise_label[t]:
                    noise_label[t] = np.random.randint(low = 0, high = self.num_class)
            # print(sum(torch.not_equal(noise_label, label_private[:x_private.size(0)//2])))
            # print(sum(torch.not_equal(label_private[x_private.size(0)//2:], label_private[:x_private.size(0)//2])))
            noise_label = noise_label.cuda()
        else:
            noise_label = label_private[x_private.size(0)//2:]
        
        
        x_noise = self.generator(z, noise_label) # pre_x returns the output of G before applying the activation
        if "randommix" in self.regularization_option:
            # Random mixup, mixup with random images, to force the generated images become strong backdoor
            
            if "test1" in self.regularization_option:
                # Mixup noise and training images, reverse the strength position, send mixture together with training images
                x_fake = torch.cat([torch.clip(self.regularization_strength * x_noise + x_private[:x_private.size(0)//2, :, :, :], -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test2" in self.regularization_option:
                # Mixup noise and random images, send mixture together with training images
                x_fake = torch.cat([torch.clip(x_noise + self.regularization_strength * torch.randn_like(x_noise).cuda(), -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test3" in self.regularization_option: # default option but removing clipping
                x_fake = torch.cat([x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test4" in self.regularization_option or "test7" in self.regularization_option or "test8" in self.regularization_option: # All mix

                #num of gout mix with training:
                num_mix1 = x_private.size(0)//4
                #number of gout mix with gout*
                num_mix2 = x_private.size(0)//2 - x_private.size(0)//4

                x_fake = torch.cat([torch.clip(x_noise[:num_mix1, :, :, :] + self.regularization_strength * x_private[:num_mix1, :, :, :], -1, 1), torch.clip(x_noise[num_mix1:, :, :, :] + self.regularization_strength * x_noise[:num_mix2, :, :, :].detach(), -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test9" in self.regularization_option:
                #num of gout mix with training:
                num_mix1 = x_private.size(0)//4
                #number of gout mix with gout*
                num_mix2 = x_private.size(0)//2 - x_private.size(0)//4

                x_fake = torch.cat([torch.clip(self.regularization_strength * x_noise[:num_mix1, :, :, :] + x_private[:num_mix1, :, :, :], -1, 1), torch.clip(self.regularization_strength * x_noise[num_mix1:, :, :, :] + x_noise[:num_mix2, :, :, :].detach(), -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test10" in self.regularization_option:
                x_fake = torch.cat([torch.clip(x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            elif "test11" in self.regularization_option or "test13" in self.regularization_option :
                # Mixup noise and training images, send mixture together with training images
                x_fake = torch.cat([(1 - self.regularization_strength) * x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
            else: # default option
                # Mixup noise and training images, send mixture together with training images
                x_fake = torch.cat([torch.clip(x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                
            label_private = torch.cat([noise_label, label_private[x_private.size(0)//2:]], dim = 0)
        else:
            # send both noise and training images
            x_fake = torch.cat([x_noise, x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
        




        # record generator_output during training
        if epoch % 20 == 0 and batch == 0:
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            imgGen = x_noise.clone()
            imgGen = denormalize(imgGen, self.dataset)
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_gout_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

            if "randommix" in self.regularization_option:
                imgGen2 = (self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :]).clone()
                imgGen2 = denormalize(imgGen2, self.dataset)
                torchvision.utils.save_image(imgGen2, train_output_path + '/{}/out_randomix_additive_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

            imgGen1 = x_fake.clone()
            imgGen1 = denormalize(imgGen1, self.dataset)
            torchvision.utils.save_image(imgGen1, train_output_path + '/{}/out_finalout_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

            imgGen3 = x_private.clone()
            imgGen3 = denormalize(imgGen3, self.dataset)
            torchvision.utils.save_image(imgGen3, train_output_path + '/{}/out_trainimg_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))

        z_private = self.model.local_list[client_id](x_fake)
        if "reduce_grad_freq" in self.regularization_option and batch % 2 == 1:
            z_private = z_private.detach()
        
        output = self.model.cloud(z_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss
        
        if "gradient_noise" in self.regularization_option:
            # see https://medium.com/analytics-vidhya/pytorch-hooks-5909c7636fb
            h = z_private.register_hook(lambda grad: grad + 1e-2 * torch.max(torch.abs(grad)) * torch.rand_like(grad).cuda())

        self.generator_optimizer.zero_grad()
        
        total_loss.backward()
        
        self.generator_optimizer.step()

        if "gradient_noise" in self.regularization_option:
            h.remove()
        
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        if "surrogate" in self.regularization_option:
            self.surrogate_model.local = self.model.local_list[client_id]
            self.surrogate_model.cloud.train()
            
            if "test7" in self.regularization_option: # All mix  train surrogate using gout, train_img and mixture
                #num of gout mix with training:
                num_mix1 = x_private.size(0)//4
                #number of gout mix with gout*
                num_mix2 = x_private.size(0)//2 - x_private.size(0)//4

                surrogate_input = torch.cat([x_noise, torch.clip(x_noise[:num_mix1, :, :, :] + self.regularization_strength * x_private[:num_mix1, :, :, :], -1, 1), torch.clip(x_noise[num_mix1:, :, :, :] + self.regularization_strength * x_noise[:num_mix2, :, :, :].detach(), -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = torch.cat([noise_label, label_private], dim = 0)
            elif "test8" in self.regularization_option: # All mix  train surrogate using gout, train_img without mixture
                surrogate_input = torch.cat([x_noise, x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = label_private
            elif "test9" in self.regularization_option: # All mix  train surrogate using gout, train_img and mixture
                #num of gout mix with training:
                num_mix1 = x_private.size(0)//4
                #number of gout mix with gout*
                num_mix2 = x_private.size(0)//2 - x_private.size(0)//4

                surrogate_input = torch.cat([x_noise, torch.clip(x_noise[:num_mix1, :, :, :] + 0.5 * x_private[:num_mix1, :, :, :], -1, 1), torch.clip(x_noise[num_mix1:, :, :, :] + 0.5 * x_noise[:num_mix2, :, :, :].detach(), -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = torch.cat([noise_label, label_private], dim = 0)
            elif "test6" in self.regularization_option: # proper mix train surrogate using gout, train_img without mixture.
                surrogate_input = torch.cat([x_noise, x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = label_private
            elif "test5" in self.regularization_option: # proper mix train surrogate using gout, train_img and mixture.
                surrogate_input = torch.cat([x_noise, torch.clip(x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], -1, 1), x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
                surrogate_label = torch.cat([noise_label, label_private], dim = 0)
            elif "test10" in self.regularization_option:
                x_fake = torch.cat([torch.clip(x_noise + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :], -1, 1), x_noise], dim = 0)
                surrogate_input = x_fake
                surrogate_label = label_private
            else:
                surrogate_input = x_fake
                surrogate_label = label_private


            with torch.no_grad():
                suro_act = self.surrogate_model.local(surrogate_input.detach())
            suro_output = self.surrogate_model.cloud(suro_act)
            suro_loss = criterion(suro_output, surrogate_label)

            self.suro_optimizer.zero_grad()
            suro_loss.backward()
            self.suro_optimizer.step()

            del suro_loss
        
        
        # zero out local model gradients to avoid unecessary poisoning effect
        if "nopoison" in self.regularization_option:
            zeroing_grad(self.model.local_list[client_id])   
         
        
        if "print_margin_stats"  in self.regularization_option and batch == 0:
            if not os.path.isdir(self.save_dir + "/margin_stats"):
                    os.makedirs(self.save_dir + "/margin_stats")
            if "surrogate" in self.regularization_option:
                confidence_score_margin_surrogate = self.calculate_margin(surrogate_input.detach(), using_surrogate_if_available=True)/surrogate_input.size(0)  # margin to the generator
                file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate.txt", "a")
                file1.write(f"{confidence_score_margin_surrogate}, ")
                file1.close()
                
                confidence_score_margin_surrogate = self.calculate_margin(x_private, using_surrogate_if_available=True)/x_private.size(0) # margin to the generator
                file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate_on_img.txt", "a")
                file1.write(f"{confidence_score_margin_surrogate}, ")
                file1.close()

                confidence_score_margin_surrogate = self.calculate_margin(x_noise.detach(), using_surrogate_if_available=True)/x_noise.size(0) # margin to the generator
                file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate_on_noise.txt", "a")
                file1.write(f"{confidence_score_margin_surrogate}, ")
                file1.close()

                if "randommix" in self.regularization_option:
                    confidence_score_margin_surrogate = self.calculate_margin(torch.clip(x_noise.detach() + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :].detach(), -1, 1), using_surrogate_if_available=True) / x_noise.size(0) # margin to the generator
                    file1 = open(f"{self.save_dir}/margin_stats/margin_surrogate_on_mixture.txt", "a")
                    file1.write(f"{confidence_score_margin_surrogate}, ")
                    file1.close()

            confidence_score_margin_target = self.calculate_margin(x_fake.detach(), using_surrogate_if_available=False)/x_fake.size(0) # margin to the generator
            file2 = open(f"{self.save_dir}/margin_stats/margin_target.txt", "a")
            file2.write(f"{confidence_score_margin_target}, ")
            file2.close()

            confidence_score_margin_target = self.calculate_margin(x_private, using_surrogate_if_available=False)/x_private.size(0) # margin to the generator
            file2 = open(f"{self.save_dir}/margin_stats/margin_target_on_img.txt", "a")
            file2.write(f"{confidence_score_margin_target}, ")
            file2.close()

            confidence_score_margin_target = self.calculate_margin(x_noise.detach(), using_surrogate_if_available=False)/x_noise.size(0) # margin to the generator
            file2 = open(f"{self.save_dir}/margin_stats/margin_target_on_noise.txt", "a")
            file2.write(f"{confidence_score_margin_target}, ")
            file2.close()

            if "randommix" in self.regularization_option:
                confidence_score_margin_target = self.calculate_margin(torch.clip(x_noise.detach() + self.regularization_strength * x_private[:x_private.size(0)//2, :, :, :].detach(), -1, 1), using_surrogate_if_available=False) / x_noise.size(0) # margin to the generator
                file2 = open(f"{self.save_dir}/margin_stats/margin_target_on_mixture.txt", "a")
                file2.write(f"{confidence_score_margin_target}, ")
                file2.close()

        return total_losses, f_losses
    
    def fidelity_test(self, client_id = 0):
        """
        Run fidelity evaluation
        """
        fidel_score = AverageMeter()
        val_loader = self.pub_dataloader
        if self.arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.Resize(224),
                transforms.CenterCrop(224))
        else:
            dl_transforms = None

        self.model.local_list[client_id].cuda()
        self.model.local_list[client_id].eval()
        self.model.cloud.cuda()
        self.model.cloud.eval()
        self.surrogate_model.local.cuda()
        self.surrogate_model.local.eval()
        self.surrogate_model.cloud.cuda()
        self.surrogate_model.cloud.eval()
        
        for i, (input, target) in enumerate(val_loader):
            if dl_transforms is not None:
                input = dl_transforms(input)
            input = input.cuda()
            target = target.cuda()
            # compute output
            with torch.no_grad():

                output = self.surrogate_model(input)
                output_target = self.model(input)

            output = output.float()
            output_target = output_target.float()

            # measure accuracy and record loss
            fidel = fidelity(output.data, output_target.data)[0]
            fidel_score.update(fidel.item(), input.size(0))
        self.logger.debug('Test (surrogate):\t' 'Fidelity {top1.val:.3f} ({top1.avg:.3f})'.format(top1=fidel_score))
        self.logger.debug(' * Fidelity {top1.avg:.3f}'.format(top1=fidel_score))
        return fidel_score.avg


    
    
    
    def generator_margin_eval(self, using_surrogate_if_available = False):
        #Distance to Decision Boundary Metrics: 
        #Various metrics exist to estimate the distance between a sample and the decision boundary of a model. 
        #One popular method is to compute the margin:
        #which measures the difference between the scores of the predicted class and the second-highest scoring class. 
        mean_margin = 0.0
        self.generator.eval()
        self.generator.cuda()

        self.model.local_list[0].eval()
        self.model.cloud.eval()

        if "surrogate" in self.regularization_option:
            self.surrogate_model.eval()
        
        margin_by_class_list = []
        # each class sample 100 images. for 10 times total
        n_rounds = 10
        for i in range(self.num_class):

            total_margin = 0.0
            for j in range(n_rounds):
                if "multi" not in self.regularization_option:
                    z = torch.randn((100, self.nz)).cuda()
                    labels = i * torch.ones([100, ]).long().cuda()
                else:
                    z = torch.randn((100//len(self.generator.generator_list), self.nz)).cuda()
                    labels = i * torch.ones([100//len(self.generator.generator_list), ]).long().cuda()
                
                
                with torch.no_grad():
                    #Get fake image from generator
                    if "multi" not in self.regularization_option:
                        fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation
                    else:
                        fake_list = []
                        for j in range(len(self.generator.generator_list)):
                            fake_list.append(self.generator.generator_list[j](z, labels))
                        fake = torch.cat(fake_list, dim = 0)
                    
                    confidence_score_margin = self.calculate_margin(fake, using_surrogate_if_available)

                total_margin += confidence_score_margin
                # print(total_margin)
            
            margin_by_class_list.append(total_margin.item()/n_rounds/100)

        mean_margin = sum(margin_by_class_list) / len(margin_by_class_list)
        return mean_margin


    def calculate_margin(self, input, using_surrogate_if_available = False):
        #Feed fake images to the classifier
        with torch.no_grad():
            if "surrogate" in self.regularization_option and using_surrogate_if_available:
                output = self.surrogate_model(input)
            else:
                act = self.model.local_list[0](input)
                output = self.model.cloud(act)
        
        #Get the Confidence Score
        confidence_score_vector = F.softmax(output) # [100, 10]
        
        # Get the maximum Confidence Score Value

        top2_score_vector = torch.topk(confidence_score_vector, k = 2, dim = 1)
        # print(top2_score_vector)

        confidence_score_margin = torch.sum(top2_score_vector[0][:, 0] - top2_score_vector[0][:, 1]) #0.0 #[100, 1]

        return confidence_score_margin

    def generator_eval(self):
        # test the variation of generator output,
        # low variation indicates a worse mode collapse and vice versa.
        self.generator.eval()
        self.generator.cuda()
        
        train_output_path = "{}/generator_final".format(self.save_dir)


        image_by_class_list = []
        # each class sample 100 images. for 10 times total
        n_rounds = 10
        for i in range(self.num_class):

            imgGen_list = []
            for j in range(n_rounds):
                if "multi" not in self.regularization_option:
                    z = torch.randn((100, self.nz)).cuda()
                    labels = i * torch.ones([100, ]).long().cuda()
                else:
                    z = torch.randn((100//len(self.generator.generator_list), self.nz)).cuda()
                    labels = i * torch.ones([100//len(self.generator.generator_list), ]).long().cuda()
                
                #Get fake image from generator
                with torch.no_grad():
                    if "multi" not in self.regularization_option:
                        fake = self.generator(z, labels).cpu() # pre_x returns the output of G before applying the activation
                    else:
                        fake_list = []
                        for j in range(len(self.generator.generator_list)):
                            fake_list.append(self.generator.generator_list[j](z, labels))
                        fake = torch.cat(fake_list, dim = 0).cpu()
                    # fake = fake * self.regularization_strength
                fake = denormalize(fake, self.dataset)
            
                if j == 0:
                    if not os.path.isdir(train_output_path):
                        os.mkdir(train_output_path)
                    if not os.path.isdir(train_output_path + "/{}".format(self.n_epochs)):
                        os.mkdir(train_output_path + "/{}".format(self.n_epochs))
                    torchvision.utils.save_image(fake, train_output_path + '/{}/out_{}.jpg'.format(self.n_epochs,"final_label{}".format(i)))
                
                imgGen_list.append(fake)
            
            imgGen = torch.cat(imgGen_list, dim = 0).clone()
            
            image_by_class_list.append(imgGen)

        
        
        
        variance_by_class_list = []
        for i in range(self.num_class):
            pass
            variance = torch.var(image_by_class_list[i].view(image_by_class_list[i].size(0), -1), dim = 1)
            # print(variance.size())
            variance = torch.mean(variance)
            variance_by_class_list.append(variance)

        #mean_variance_by_class
        mean_var = np.mean(variance_by_class_list)
        print(f"variance_by_class_list is {variance_by_class_list}")
        # print(f"mean_var is {mean_var}")
        return mean_var.item()

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