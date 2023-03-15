import torch
import numpy as np
import torch.nn as nn
from torch.serialization import save
import torchvision.transforms as transforms
import models.architectures_torch as architectures
from utils import setup_logger, accuracy, fidelity, set_bn_eval, student_Loss, AverageMeter, WarmUpLR, apply_transform_test, apply_transform, TV, l2loss, dist_corr, get_PSNR, zeroing_grad
from utils import freeze_model_bn, average_weights, DistanceCorrelationLoss, spurious_loss, prune_top_n_percent_left, dropout_defense, prune_defense
from thop import profile
import logging
from torch.autograd import Variable
from models import get_model
import pytorch_ssim
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from datetime import datetime
import os, copy
import time
from shutil import rmtree
from datasets import get_dataset
from models import get_model
from tqdm import tqdm
import glob

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def steal_test(self, model, val_loader, attack_client = 1):
    """
    Run evaluation
    """
    # batch_time = AverageMeter()
    fidel_score = AverageMeter()
    top1 = AverageMeter()

    if not model.cloud_classifier_merge:
        model.merge_classifier_cloud()

    model.local_list[0].cuda()
    model.local_list[0].eval()
    model.cloud.cuda()
    model.cloud.eval()
    
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


def adversarial_attack(self, attack_option, attack_client = 0, e = None):
    total_succ = 0
    self.surrogate_model.local.eval()
    self.surrogate_model.cloud.eval()
    criterion = nn.CrossEntropyLoss()
    if attack_option == "PGD_target":
        
        iter_max = 10
        if e is None:
            e = 0.05
        for temporal in range(iter_max):
            org_target = temporal
            attack_target = (org_target + 5)%9
            # history_list.append(org_target)
            self.logger.debug(f"###Round {temporal} src_label: {org_target} target_label: {attack_target}")
            
            
            succ = 0
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            
            fake_label = torch.LongTensor(1)
            fake_label[0] = attack_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])

            diff_succ = 0.0
            diff_all  = 0.0

            for i in range(image_arr.shape[0]):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.model.local_list[attack_client](org_image)
                output = self.model.cloud(act)
                # print(output)
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                fake_image = org_image.clone()
                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                for iter in range(50):
                    # calculate gradient
                    grad = torch.zeros(1, 3, 32, 32).cuda()
                    fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)
                    if fake_image.grad is not None:
                        fake_image.grad.zero_()
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)
                    loss = criterion(output, fake_label)
                    loss.backward()
                    #print(loss)
                    grad += torch.sign(fake_image.grad)

                    fake_image = fake_image - grad * e
                    fake_image[fake_image > max_val] = max_val
                    fake_image[fake_image < min_val] = min_val
                    act = self.model.local_list[attack_client](fake_image)
                    output = self.model.cloud(act)
                    _, fake_pred = output.topk(1, 1, True, True)
                    fake_pred = fake_pred[0, 0]
                    
                    if fake_label.item() == fake_pred.item() or iter == 49:
                        # print(fake_pred.item(), fake_label.item())
                        attack_pred_list = []
                        act = self.surrogate_model.local(fake_image)
                        output = self.surrogate_model.cloud(act)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                        # if (i + 1) % 20 == 0:
                        #     print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                        #     '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\titer: ' + str(iter) + '\tsucc: ' + str(succ))

                        org_label[i] = org_pred.item()
                        attack_label[i] = fake_pred.item()
                        succ_iter[i] = iter + 1
                        
                        diff = torch.sum((org_image - fake_image) ** 2).item()
                        diff_all += diff

                        if fake_label.item() == fake_pred:
                            diff_succ += diff
                            succ += 1
                            succ_label[i] = 1
                        break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            
            str_log = 'src: ' + str(org_target) + '\ttar: ' + str(attack_target)+ '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")
    elif attack_option == "FGSM":
        if e is None:
            e = 0.1
        iter_max = 10
        for temporal in range(iter_max):
            org_target = temporal
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            succ = 0
            diff_succ = 0.0
            diff_all  = 0.0
            fake_label = torch.LongTensor(1)
            fake_label[0] = org_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])
            for i in range(image_arr.shape[0]):
                #for i in range(2):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.model.local_list[attack_client](org_image)
                output = self.model.cloud(act)
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]

                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                
                fake_image = org_image.clone()

                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                
                # calculate gradient
                grad = torch.zeros(1, 3, 32, 32).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

                if fake_image.grad is not None:
                    fake_image.grad.zero_()
                act = self.surrogate_model.local(fake_image)
                output = self.surrogate_model.cloud(act)
                loss = criterion(output, fake_label)
                loss.backward()
                grad -= torch.sign(fake_image.grad)

                fake_image = fake_image - grad * e # e is epsilon in FGSM:  https://arxiv.org/pdf/1706.06083.pdf
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                act = self.model.local_list[attack_client](fake_image)
                output = self.model.cloud(act)

                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred.data[0, 0]

                attack_pred_list = []
                act = self.surrogate_model.local(fake_image)
                output = self.surrogate_model.cloud(act)
                _, attack_pred = output.topk(1, 1, True, True)
                attack_pred_list.append(attack_pred.data[0, 0].item())

                if (i + 1) % 20 == 0:
                    print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                        '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\tsucc: ' + str(succ))

                org_label[i] = org_pred.item()
                attack_label[i] = fake_pred.item()
                
                diff = torch.sum((org_image - fake_image) ** 2).item()
                diff_all += diff

                if fake_label.item() != fake_pred:
                    diff_succ += diff
                    succ += 1
                    succ_label[i] = 1
                    # break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            
            str_log = 'src: ' + str(org_target) + '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            # print(str_log)
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")
    elif attack_option == "PGD":
        if e is None:
            e = 0.02
        for temporal in range(self.num_class):
            org_target = temporal
            image_arr = torch.load('saved_tensors/data100/class_'+ str(org_target) +'.pth.tar')
            succ = 0
            diff_succ = 0.0
            diff_all  = 0.0
            fake_label = torch.LongTensor(1)
            fake_label[0] = org_target
            fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
            org_label = torch.zeros(image_arr.shape[0])
            attack_label = torch.zeros(image_arr.shape[0])
            succ_label = torch.zeros(image_arr.shape[0])
            succ_iter = torch.zeros(image_arr.shape[0])
            for i in range(image_arr.shape[0]):
            #for i in range(2):
                org_image = torch.FloatTensor(1, 3, 32, 32)
                org_image[0] = image_arr[i]
                # print(image_arr[i])
                org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

                act = self.surrogate_model.local(org_image)
                output = self.surrogate_model.cloud(act)

                # activation = nn.LogSoftmax(1)

                # output = activation(output)
                
                _, org_pred = output.topk(1, 1, True, True)
                org_pred = org_pred.data[0, 0]
                # if i < 50:
                #     print(org_pred)
                fake_image = org_image.clone()

                #modify the original image
                max_val = torch.max(org_image).item()
                min_val = torch.min(org_image).item()
                for iter in range(50): # PGD: 
                    # calculate gradient
                    grad = torch.zeros(1, 3, 32, 32).cuda()
                    fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

                    if fake_image.grad is not None:
                        fake_image.grad.zero_()
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)
                    loss = criterion(output, fake_label)
                    loss.backward()
                    #print(loss)
                    grad -= torch.sign(fake_image.grad)

                    fake_image = fake_image - grad * e # e is alpha in PGD:  https://arxiv.org/pdf/1706.06083.pdf
                    fake_image[fake_image > max_val] = max_val
                    fake_image[fake_image < min_val] = min_val
                    act = self.surrogate_model.local(fake_image)
                    output = self.surrogate_model.cloud(act)

                    _, fake_pred = output.topk(1, 1, True, True)
                    fake_pred = fake_pred.data[0, 0]

                    if fake_label.item() != fake_pred.item() or iter == 49:
                        # print(fake_pred.item(), fake_label.item())
                        attack_pred_list = []

                        act = self.surrogate_model.local(fake_image)
                        output = self.surrogate_model.cloud(act)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                        if (i + 1) % 20 == 0:
                            print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                            '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(e) + '\titer: ' + str(iter) + '\tsucc: ' + str(succ))

                        org_label[i] = org_pred.item()
                        attack_label[i] = fake_pred.item()
                        succ_iter[i] = iter + 1
                        
                        diff = torch.sum((org_image - fake_image) ** 2).item()
                        diff_all += diff

                        if fake_label.item() != fake_pred:
                            diff_succ += diff
                            succ += 1
                            succ_label[i] = 1
                        break


            diff_all /= (1.0 * image_arr.shape[0])
            if succ > 0:
                diff_succ /= (1.0 * succ)
            #print('total: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()))
            
            str_log = 'src: ' + str(org_target) + '\ttar: ' + '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
            self.logger.debug(str_log)
            total_succ += succ
        self.logger.debug(f"Total succeed times {total_succ}.")



def steal_attack(self, num_query = 10, num_epoch = 200, attack_client=0, attack_style = "TrainME_option", 
                data_proportion = 0.2, noniid_ratio = 1.0, train_clas_layer = -1, surrogate_arch = "same", adversairal_attack_option = False):
    
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
    
    start_time = time.time()
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

        if "noisew" in attack_style:
            self.noise_w = float(attack_style.split("noisew")[1])
        else:
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
        
        if "last" in attack_style:
            last_n_batch = int(attack_style.split("last")[-1])
        else:
            last_n_batch = 99
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
        
        if "last" in attack_style:
            last_n_batch = int(attack_style.split("last")[-1])
        else:
            last_n_batch = 99
        soft_alpha = 0.9

        if "alpha" in attack_style:
            soft_alpha = float(attack_style.split("alpha")[-1])

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
    

    if GM_option and resume_option: #TODO cancel this
        if self.arch == "vgg11_bn":
            self.surrogate_model.resplit(9)
            self.model.resplit(9)
        elif self.arch == "resnet20":
            self.surrogate_model.resplit(7)
            self.model.resplit(7)
        elif self.arch == "mobilenetv2":
            self.surrogate_model.resplit(13)
            self.model.resplit(13)
    if SoftTrain_option and resume_option: #TODO cancel this
        if self.arch == "vgg11_bn":
            self.surrogate_model.resplit(9)
            self.model.resplit(9)
        elif self.arch == "resnet20":
            self.surrogate_model.resplit(7)
            self.model.resplit(7)
        elif self.arch == "mobilenetv2":
            self.surrogate_model.resplit(13)
            self.model.resplit(13)

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
            attacker_loader_list, _, _ = get_cifar10_trainloader(batch_size=100, num_workers=0, shuffle=True, num_client=int(1/data_proportion), noniid_ratio = noniid_ratio)
        elif self.dataset == "imagenet":
            attacker_loader_list = get_imagenet_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion), noniid_ratio = noniid_ratio)
        elif self.dataset == "svhn":
            attacker_loader_list, _, _ = get_SVHN_trainloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
        elif self.dataset == "mnist":
            attacker_loader_list, _= get_mnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
        elif self.dataset == "fmnist":
            attacker_loader_list, _= get_fmnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
        elif self.dataset == "femnist":
            attacker_loader_list, _= get_femnist_bothloader(batch_size=100, num_workers=4, shuffle=True, num_client=int(1/data_proportion))
        else:
            raise("Unknown Dataset!")

    attacker_dataloader = attacker_loader_list[attack_client]
    
    # print(attacker_dataloader)
    atk_transforms = torch.nn.Sequential(
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    )
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
                
            max_image_id = 0
            for file in glob.glob(saved_crafted_image_path + "*"):
                if "image" in file and "grad_image" not in file:
                    image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                    if image_id > max_image_id:
                        max_image_id = image_id
        
            for file in glob.glob(saved_crafted_image_path + "*"):
                if "image" in file and "grad_image" not in file:
                    
                    saved_image = torch.load(file)
    
                    image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))

                    if image_id > max_image_id - last_n_batch: # collect only the last two valid data batch. (even this is very bad)
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
            cumulated_query = 0
            while True:
                if cumulated_query > num_query: # in query budget, craft soft label 
                    break
                for i, (images, labels) in enumerate(attacker_dataloader):
                    self.optimizer_zero_grad()
                    if "aug" in attack_style:
                        images = atk_transforms(images)
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
                    cumulated_query += self.num_class * 100
                    if cumulated_query > num_query: # in query budget, craft soft label 
                        break
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

                    # else:  # out of query budget, use hard label
                    #     derived_label = F.one_hot(labels, num_classes=self.num_class)

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
            print(f"max_image is {max_image_id}")
            
            for file in glob.glob(saved_crafted_image_path + "*"):
                if "image" in file and "grad_image" not in file:
                    
                    for label in range(self.num_class): # put same label together
                        image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))

                        if image_id > max_image_id - last_n_batch: # collect only the last two valid data batch. (even this is very bad)
                            try:
                                saved_image = torch.load(file)
                                saved_grads = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{label}.pt")
                                saved_label = label * torch.ones(saved_grads.size(0), ).long()
                                # saved_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")

                                save_images.append(saved_image.clone())
                                save_grad.append(saved_grads.clone()) # add a random existing grad.
                                save_label.append(saved_label.clone())
                            except:
                                break
        else:
            self.model.local_list[attack_client].eval()
            self.model.cloud.eval()
            for i, (inputs, target) in enumerate(knockoff_loader):
                if i * self.num_class * 100 >= num_query: # limit grad query budget
                    break
                if "aug" in attack_style:
                    inputs = atk_transforms(inputs)
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

    # if SoftTrain_option or TrainME_option: #TODO: reverse this.
    #     dl_transforms = None
    
    time_cost = time.time() - start_time

    self.logger.debug(f"Time cost on preparing the attack: {time_cost}")

    start_time = time.time()

    min_grad_loss = 9.9
    acc_loss_min_grad_loss = 9.9
    val_acc_max = 0.0
    fidelity_max = 0.0
    
    best_tail_state_dict = None
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
            
            if SoftTrain_option: #perform matching (distable augmentation)
                for idx, (index, image, grad, label) in enumerate(dl):
                    image = image.cuda()
                    grad = grad.cuda()
                    label = label.cuda()
                    self.suro_optimizer.zero_grad()
                    act = self.surrogate_model.local(image)
                    output = self.surrogate_model.cloud(act)
                    _, real_label = label.max(dim = 1)
                    ce_loss = (1 - soft_lambda) * criterion(output, real_label) + soft_lambda * torch.mean(torch.sum(-label * torch.nn.functional.log_softmax(output, dim=1), dim=1))
                    # ce_loss = torch.mean(torch.sum(-label * torch.nn.functional.log_softmax(output, dim=1), dim=1))
                    total_loss = ce_loss
                    total_loss.backward()
                    self.suro_optimizer.step()
            
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
                    ce_loss = criterion(output, real_label)
                else:
                    ce_loss = criterion(output, label)

                total_loss = ce_loss
                total_loss.backward()
                self.suro_optimizer.step()

                acc_loss_list.append(ce_loss.detach().cpu().item())

                if Knockoff_option or SoftTrain_option:
                    _, real_label = label.max(dim=1)
                    acc = accuracy(output.data, real_label)[0]
                else:
                    acc = accuracy(output.data, label)[0]
                    
                acc_list.append(acc.cpu().item())
            
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
        
        if GM_option:
            grad_loss_mean = np.mean(grad_loss_list)
        else:
            grad_loss_mean = None
        min_grad_loss = None
        acc_loss_mean = np.mean(acc_loss_list)
        avg_acc = np.mean(acc_list)

        val_accu, fidel_score = self.steal_test(attack_client=attack_client)

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
    

    if gradient_matching and TrainME_option:

        # for epoch in range(num_epoch, num_epoch + 100):
        for epoch in range(0, 100):
            grad_loss_list = []
            acc_loss_list = []
            acc_list = []
            self.suro_scheduler.step(epoch)
            if resume_option and TrainME_option:
                acc_loss_list.append(0.0)
                acc_list.append(0.0)
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
            elif TrainME_option:

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

            grad_loss_mean = np.mean(grad_loss_list)
            val_accu, fidel_score = self.steal_test(attack_client=attack_client)
            acc_loss_mean = None
            avg_acc = None
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
            self.logger.debug("GMepoch: {}, train_acc: {}, val_acc: {}, fidel_score: {}, acc_loss: {}, grad_loss: {}".format(epoch, avg_acc, val_accu, fidel_score, acc_loss_mean, grad_loss_mean))

    time_cost = time.time() - start_time

    self.logger.debug(f"Time cost on training surrogate model: {time_cost}")


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

    if adversairal_attack_option:
        self.adversarial_attack("PGD_target", attack_client=attack_client, e = 0.02)
        self.adversarial_attack("FGSM", attack_client=attack_client, e = 0.1)
        self.adversarial_attack("PGD", attack_client=attack_client, e = 0.001)


    