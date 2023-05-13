import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import models.architectures_torch as architectures
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
import wandb

class Trainer:
    def __init__(self, arch, cutting_layer, batch_size, n_epochs, scheme="V2", num_client=2, dataset="cifar10",
                 logger=None, save_dir=None, regularization_option="None", regularization_strength=0, learning_rate=0.1, 
                 random_seed=123, load_from_checkpoint = False, attack_confidence_score = False, num_freeze_layer = 0,
                 finetune_freeze_bn = False, load_from_checkpoint_server = False, source_task = "cifar100", client_sample_ratio = 1.0,
                 save_activation_tensor = False, noniid = 1.0, last_client_fix_amount = -1, attacker_querying_budget_num_step = -1):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.arch = arch
        self.batch_size = batch_size
        self.lr = learning_rate
        self.finetune_freeze_bn = finetune_freeze_bn
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

            if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
                self.nz = 512
                self.generator = architectures.GeneratorC(nz=self.nz, num_classes = self.num_class, ngf=128, nc=self.image_shape[0], img_size=self.image_shape[2])
                self.generator.cuda()
                self.local_params.append(self.generator.parameters())

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
                if epoch > self.warm:
                    self.scheduler_step(epoch)
                
                if self.client_sample_ratio  == 1.0:
                    idxs_users = range(self.num_client)
                else:
                    idxs_users = np.random.choice(range(self.actual_num_users), self.num_client, replace=False) # 10 out of 1000
                    
                # sample num_client for parapllel training from actual number of users, take their iterator as well.
                client_iterator_list = []
                for client_id in range(self.num_client):
                    if ("gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option) and idxs_users[client_id] == self.attacker_client_id:
                        client_iterator_list.append(None)
                    else:
                        client_iterator_list.append(saved_iterator_list[idxs_users[client_id]])

                
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
                    
                    
                    for client_id in idxs_users:
                        # Get data TODO: solve this fatal error
                        if client_id != self.attacker_client_id: # if current client is not the attack client (default is the last one)
                            try:
                                images, labels = next(client_iterator_list[client_id])
                                if images.size(0) != self.batch_size:
                                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                                    images, labels = next(client_iterator_list[client_id])
                            except StopIteration:
                                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                                images, labels = next(client_iterator_list[client_id])

                            if dl_transforms is not None:
                                images = dl_transforms(images)
                        else: # if the client is the attacker client:
                            images, labels = self.get_data_MEA_client(client_iterator_list, client_id, epoch, dl_transforms)
                        
                        if self.scheme == "V2":
                            self.optimizer_zero_grad()
                        
                        
                        if client_id != self.attacker_client_id:
                            # Train step (client/server)
                            train_loss, f_loss = self.train_target_step(images, labels, client_id, epoch, batch)
                        else: 
                            # MEA client, perform training, collecting gradients using the adv data
                            if epoch > self.attack_start_epoch:
                                
                                if "craft_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.craft_train_target_step(client_id)
                                elif "gan_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.gan_train_target_step(client_id, self.batch_size, epoch, batch)
                                elif "gan_assist_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.gan_assist_train_target_step(images, labels, client_id, epoch, batch)
                                elif "GM_train_ME" in self.regularization_option or "soft_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option:
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id, epoch, batch, save_grad = True, skip_regularization=True) # adv clients won't comply to the defense
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id, epoch, batch)
                            else:
                                if "gan_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option:
                                    pass # do nothing prior to the starting epoch   
                                elif "soft_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option: # adv clients won't comply to the defense
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id, epoch, batch, skip_regularization=True)
                                else:
                                    train_loss, f_loss = self.train_target_step(images, labels, client_id, epoch, batch)

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
                    
                # Step the warmup scheduler
                if epoch <= self.warm:
                    self.scheduler_step(warmup=True)

                # Validate and get average accu among clients
                avg_accu = 0
                avg_accu, loss = self.validate_target(client_id=0)
                wandb.log({f"val_acc": avg_accu, "val_loss": loss})
                # Save the best model
                if avg_accu > best_avg_accu:
                    self.save_model(epoch, is_best=True)
                    best_avg_accu = avg_accu

                # Save Model regularly
                if epoch % 50 == 0 or epoch == self.n_epochs or epoch in epoch_save_list:  # save model
                    self.save_model(epoch)
                    
                    if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
                        torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))
        
        #save final images for gan_train_ME generator.
        if "gan_train_ME" in self.regularization_option or "gan_assist_train_ME" in self.regularization_option:
            self.generator.cuda()
            self.generator.eval()
            z = torch.randn((10, self.nz)).cuda()
            train_output_path = "{}/generator_final".format(self.save_dir)
            for i in range(self.num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                #Get fake image from generator
                fake = self.generator(z, labels) # pre_x returns the output of G before applying the activation

                imgGen = fake.clone()
                imgGen = denormalize(imgGen, self.dataset)
                if not os.path.isdir(train_output_path):
                    os.mkdir(train_output_path)
                if not os.path.isdir(train_output_path + "/{}".format(self.n_epochs)):
                    os.mkdir(train_output_path + "/{}".format(self.n_epochs))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(self.n_epochs,"final_label{}".format(i)))
            torch.save(self.generator.state_dict(), self.save_dir + 'checkpoint_generator_{}.tar'.format(epoch))

        if not self.call_resume:
            self.logger.debug("Best Average Validation Accuracy is {}".format(best_avg_accu))
            wandb.run.summary["Best-Val-Accuracy"] = best_avg_accu
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

    def train_target_step(self, x_private, label_private, client_id, epoch, batch, save_grad = False, skip_regularization = False):
        self.model.cloud.train()
        self.model.local_list[client_id].train()
        
        if save_grad and "diffaug" in self.regularization_option:
            if "half_half" not in self.regularization_option:
                x_private = DiffAugment.DiffAugment(x_private, 'color,translation,cutout') # a parameterized augmentation module.
            else:
                x_private = torch.cat([DiffAugment.DiffAugment(x_private[:x_private.size(0)//2, :, :, :], 'color,translation,cutout'), x_private[x_private.size(0)//2:, :, :, :]], dim = 0) # a parameterized augmentation module.
        
        if save_grad and "reverse_grad" in self.regularization_option: #TODO: test this
            x_private.requires_grad = True
            def noise_hook(grad):
                return torch.multiply(grad, -1 * torch.ones_like(grad))
            h = x_private.register_hook(noise_hook)

        if save_grad: # if we save grad, meaning we are doing softTrain, which has a poisoning effect, we do not want this to affect aggregation.
            # collect gradient
            if not os.path.isdir(self.save_dir + "/saved_grads"):
                os.makedirs(self.save_dir + "/saved_grads")
            
            if "advnoise" in self.regularization_option:
                x_private.retain_grad()
                if not os.path.isdir(self.save_dir + "/pool_advs"):
                    os.makedirs(self.save_dir + "/pool_advs")
            # collect image/label
            if epoch > self.n_epochs - 10 and self.rotate_label == 0:
                max_allow_image_id = 500 # 500 times batch size is a considerable amount.
                if self.query_image_id <= max_allow_image_id:
                    torch.save(x_private.detach().cpu(), self.save_dir + f"/saved_grads/image_{self.query_image_id}.pt")
                    torch.save(label_private.detach().cpu(), self.save_dir + f"/saved_grads/label_{self.query_image_id}.pt")

        if save_grad and "randomnoise" in self.regularization_option: # add a random noise of norm value = regularization_strength to the input.
            
            x_private = x_private + self.regularization_strength * torch.randn_like(x_private)

        x_private = x_private.cuda()
        label_private = label_private.cuda()

        # Option to Freeze batchnorm parameter of the client-side model.
        if self.load_from_checkpoint and self.finetune_freeze_bn:
            freeze_model_bn(self.model.local_list[client_id])
        

        # define the hook function 
        # def get_activation_gradient(grads, name):
        #     def hook(model, input, output):
        #         grads[name] = output.grad
        #     return hook
        # grads = {}
        # self.model.first_cloud_layer.register_backward_hook(get_activation_gradient(grads, 'first_cloud_layer'))
        
        if self.arch != "ViT":
            # Final Prediction Logits (complete forward pass)
            z_private = self.model.local_list[client_id](x_private)
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
                def noise_hook(grad):
                    return torch.add(grad, 0.01 * torch.rand_like(grad).cuda())
                h = z_private.register_hook(noise_hook)
            # hook activation gradient and add noise to it.


            # if "gradient_noise_cloud" in self.regularization_option:
            #     for i, p in enumerate(self.model.cloud.parameters()):
            #         p.register_hook(lambda grad: torch.add(grad, self.regularization_strength * torch.rand_like(grad).cuda()))
            # if "gradient_noise_local" in self.regularization_option:
            #     for i, p in enumerate(self.model.local_list[0].parameters()):
            #         p.register_hook(lambda grad: torch.add(grad, self.regularization_strength * torch.rand_like(grad).cuda()))

            # if "gradient_clipping" in self.regularization_option:
            #     for i, p in enumerate(self.model.local_list[0].parameters()):
            #         p.register_hook(lambda grad: torch.clip(grad, -self.regularization_strength, self.regularization_strength))

        total_loss.backward()

        if save_grad and "reverse_grad" in self.regularization_option: #TODO: test this
            h.remove()
        
        if not skip_regularization:
            if "gradient_noise" in self.regularization_option and self.arch != "ViT":
                h.remove()

        if save_grad: # if we save grad, meaning we are doing softTrain, which has a poisoning effect, we do not want this to affect aggregation.
            if "naive_train_ME" in self.regularization_option:
                pass
            else:
                zeroing_grad(self.model.local_list[client_id])
                torch.save(z_private.grad.detach().cpu(), self.save_dir + f"/saved_grads/grad_image{self.query_image_id}_label{self.rotate_label}.pt")
            if "GM_train_ME" in self.regularization_option:
                torch.save(z_private.detach().cpu(), self.save_dir + f"/saved_grads/act_{self.query_image_id}_label{self.rotate_label}.pt")
            
            if "advnoise" in self.regularization_option:
                torch.save(x_private.detach().cpu() - self.regularization_strength * x_private.grad.detach().cpu() , self.save_dir + f"/pool_advs/lastest_batch{batch}.pt")
            
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss
        return total_losses, f_losses
    
    def craft_train_target_step(self, client_id):
        lambda_TV = 0.0
        lambda_l2 = 0.0
        craft_LR = 1e-1
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

        torch.save(fake_image.detach().cpu(), image_save_path + f'image_{self.craft_image_id}.pt')
        torch.save(fake_label.cpu(), image_save_path + f'label_{self.craft_image_id}.pt')
        torch.save(z_private.grad.detach().cpu(), image_save_path + f'grad_{self.craft_image_id}.pt')

        self.craft_step_count += 1
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

        if poison_option and batch % 2 == 1: #poison step
            label_private = torch.randint(low=0, high=self.num_class, size = [self.batch_size, ]).cuda()
            x_private = x_private.detach()
        
        if epoch % 5 == 0 and batch == 0:
            imgGen = x_private.clone()
            imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(epoch,
                                                                                            batch * self.batch_size + self.batch_size))
        
        # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
        g_noise_out_dist = torch.mean(torch.abs(x_private[:B, :] - x_private[B:, :]))
        g_noise_z_dist = torch.mean(torch.abs(z[:B, :] - z[B:, :]).view(B,-1),dim=1)
        noise_w = 50

        if "noreg" in self.regularization_option:
            noise_w = 0

        g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * noise_w

        # z_private = self.model.local_list[client_id](x_private)

        # output = self.model.cloud(z_private)
        output = self.model(x_private)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        total_loss = f_loss - g_noise

        if "gradient_noise" in self.regularization_option: #TODO: test this
            def noise_hook(grad):
                return torch.add(grad, 0.01 * torch.rand_like(grad).cuda())
            h = x_private.register_hook(noise_hook)

        total_loss.backward()

        zeroing_grad(self.model.local_list[client_id])

        if "gradient_noise" in self.regularization_option: #TODO: test this
            h.remove()

        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses

    def gan_assist_train_target_step(self, x_private, label_private, client_id, epoch, batch):

        #if enable poison option
        self.model.cloud.train()
        self.model.local_list[client_id].train()
        self.generator.cuda()
        self.generator.train()

        x_private = x_private.cuda()
        label_private = label_private.cuda()

        train_output_path = self.save_dir + "assisted_generator_train"
        if not os.path.isdir(train_output_path):
            os.makedirs(train_output_path)


        #Sample Random Noise
        # z = torch.randn((x_private.size(0), self.nz)).cuda()
        
        z_label = torch.stack([label_private, label_private]).view(-1)

        #Get class-dependent noise, adding to x_private lately
        if "half_half" in self.regularization_option:
            z = torch.randn((x_private.size(0) * 2, self.nz)).cuda()
            x_noise = self.generator(z, z_label) # pre_x returns the output of G before applying the activation
            # if "reverse_grad" in self.regularization_option: #TODO: test this
            x_noise = self.regularization_strength * x_noise
            x_fake = torch.cat([x_noise[:x_private.size(0)//2, :, :, :] + x_private[:x_private.size(0)//2, :, :, :], x_private[x_private.size(0)//2:, :, :, :]], dim = 0)
        else:
            z = torch.randn((x_private.size(0) * 2, self.nz)).cuda()
            x_noise = self.generator(z, z_label) # pre_x returns the output of G before applying the activation
            # if "reverse_grad" in self.regularization_option: #TODO: test this
            x_noise = self.regularization_strength * x_noise
            x_fake = x_noise[:x_private.size(0), :, :, :] + x_private
        
        #TODO:
        # step-1 scale the Gnoise to L norm
        # step-2 reverse the direction of the gradients on Gnoise. when backprop.
        # see how it goes.

        #TODO::
        # increase number samples from DiffAug
        if epoch % 5 == 0 and batch == 0:
            imgGen = x_noise.clone()
            imgGen = denormalize(imgGen, self.dataset)
            if not os.path.isdir(train_output_path + "/{}".format(epoch)):
                os.mkdir(train_output_path + "/{}".format(epoch))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(epoch, batch * self.batch_size + self.batch_size))
        

        output = self.model(x_fake)
        
        criterion = torch.nn.CrossEntropyLoss()

        f_loss = criterion(output, label_private)

        # bound x_noise
        # if "norm1" in self.regularization_option:
        #     lnorm_penalty = (torch.norm(x_noise, 1)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     total_loss = f_loss + lnorm_penalty
        # elif "var" in self.regularization_option:
        #     lnorm_penalty = (torch.mean(torch.std(x_noise, dim = [1,2,3])) - self.regularization_strength) ** 2
        #     total_loss = f_loss + lnorm_penalty
        # elif "norm2" in self.regularization_option:
        #     # lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     # lnorm_penalty = 0.1 * torch.norm(x_noise - 0.1, 2) # fix x_noise to be around 0.1
        #     total_loss = f_loss + lnorm_penalty
        # elif "plateau1" in self.regularization_option:
        #     # lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     # lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     lnorm_penalty = self.regularization_strength * torch.norm(x_noise - 0.1, 1) # fix x_noise to be around 0.1
        #     total_loss = f_loss + lnorm_penalty
        # elif "plateau2" in self.regularization_option:
        #     # lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     # lnorm_penalty = (torch.norm(x_noise, 2)/torch.numel(x_noise) - self.regularization_strength) ** 2
        #     lnorm_penalty = self.regularization_strength * torch.norm(x_noise - 0.1, 2) # fix x_noise to be around 0.1
        #     total_loss = f_loss + lnorm_penalty
        # else:
        total_loss = f_loss

        if "diverse" in self.regularization_option:
            # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
            g_noise_out_dist = torch.mean(torch.abs(x_noise[:x_private.size(0), :] - x_noise[x_private.size(0):, :]))
            g_noise_z_dist = torch.mean(torch.abs(z[:x_private.size(0), :] - z[x_private.size(0):, :]).view(x_private.size(0),-1),dim=1)
            noise_w = 50
            g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * noise_w
            if "reverse_grad" in self.regularization_option: #TODO: test this
                total_loss += g_noise
            else:
                total_loss -= g_noise
        

        if "reverse_grad" in self.regularization_option: #TODO: test this
            def noise_hook(grad):
                return torch.multiply(grad, -1 * torch.ones_like(grad).cuda())
            h = x_fake.register_hook(noise_hook)
        
        if "gradient_noise" in self.regularization_option: #TODO: test this
            def noise_hook(grad):
                return torch.add(grad, 0.01 * torch.rand_like(grad).cuda())
            h = x_fake.register_hook(noise_hook)

        total_loss.backward()

        # collect images and labels

        max_allow_image_id = 500 # 500 times batch size is a considerable amount.
        if self.query_image_id <= max_allow_image_id:
            if not os.path.isdir(self.save_dir + "/saved_grads"):
                os.makedirs(self.save_dir + "/saved_grads")
            torch.save(x_private.detach().cpu(), self.save_dir + f"/saved_grads/image_{self.query_image_id}.pt")
            torch.save(label_private.detach().cpu(), self.save_dir + f"/saved_grads/label_{self.query_image_id}.pt")


        if "gradient_noise" in self.regularization_option: #TODO: test this
            h.remove()
        if "reverse_grad" in self.regularization_option: #TODO: test this
            h.remove()
        
        total_losses = total_loss.detach().cpu().numpy()
        f_losses = f_loss.detach().cpu().numpy()
        del total_loss, f_loss

        return total_losses, f_losses



    




    def save_model(self, epoch, is_best=False):
        if is_best:
            epoch = "best"
        torch.save(self.model.local_list[0].state_dict(), self.save_dir + 'checkpoint_client_{}.tar'.format(epoch))
        torch.save(self.model.cloud.state_dict(), self.save_dir + 'checkpoint_cloud_{}.tar'.format(epoch))

    

    def get_data_MEA_client(self, client_iterator_list, client_id, epoch, dl_transforms):

        self.old_image = None
        self.old_label = None

        if ("gan_train_ME" in self.regularization_option or "craft_train_ME" in self.regularization_option):   # Data free no data.
            
            return None, None
        
        elif ("soft_train_ME" in self.regularization_option or "GM_train_ME" in self.regularization_option) and epoch > self.attack_start_epoch:  # SoftTrain, attack mode
            
            if self.rotate_label == self.num_class or (self.query_image_id == 0 and self.rotate_label == 0):
                
                if (self.query_image_id == 0 and self.rotate_label == 0):
                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                
                if self.rotate_label == self.num_class:
                    self.rotate_label = 0
                    self.query_image_id += 1
                
                try:
                    self.old_images, self.old_labels = next(client_iterator_list[client_id])
                    self.old_images = dl_transforms(self.old_images)
                except StopIteration:
                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                    self.old_images, self.old_labels = next(client_iterator_list[client_id])
                    self.old_images = dl_transforms(self.old_images)
                
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
                images, labels = next(client_iterator_list[client_id])
                if images.size(0) != self.batch_size:
                    client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                    images, labels = next(client_iterator_list[client_id])
            except StopIteration:
                client_iterator_list[client_id] = iter(self.client_dataloader[client_id])
                images, labels = next(client_iterator_list[client_id])

            if dl_transforms is not None:
                images = dl_transforms(images)
            
            if epoch > self.n_epochs - 10:
                if ("gan_assist_train_ME" in self.regularization_option or "naive_train_ME" in self.regularization_option):
                    self.query_image_id += 1
            
            
            return images, labels