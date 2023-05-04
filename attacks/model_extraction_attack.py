import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import models.architectures_torch as architectures
from utils import accuracy, fidelity, AverageMeter, TV, l2loss, setup_logger
import logging
import torch.nn.functional as F
import torchvision
import os, copy
import time
from shutil import rmtree
from datasets import get_dataset, denormalize, get_image_shape
import glob
import wandb
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def train_generator(logger, save_dir, target_model, generator, target_dataset_name, num_class, num_query, nz, resume = False, assist = False):
        
    lr_G = 2e-4
    num_steps = num_query // 100 # train once
    steps = sorted([int(step * num_steps) for step in [0.1, 0.3, 0.5]])
    scale = 3e-1
    
    D_w = 50 # noise_w

    # if target_dataset_name == "cifar10":
    #     D_w = D_w * 10
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G )
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, steps, scale)
    
    train_output_path = save_dir + "/generator_train"
    if os.path.isdir(train_output_path):
        rmtree(train_output_path)
    os.makedirs(train_output_path)
    
    if resume:
        G_state_dict = torch.load(save_dir + f"/checkpoint_generator_200.tar")
        generator.load_state_dict(G_state_dict)
        generator.cuda()
        generator.eval()

        if not assist:
            z = torch.randn((10, nz)).cuda()
            for i in range(num_class):
                labels = i * torch.ones([10, ]).long().cuda()
                #Get fake image from generator
                fake = generator(z, labels) # pre_x returns the output of G before applying the activation

                imgGen = fake.clone()
                imgGen = denormalize(imgGen, target_dataset_name)
                if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                    os.mkdir(train_output_path + "/{}".format(num_steps))
                torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))
    else:
        generator.cuda()
        generator.train()

        criterion = torch.nn.CrossEntropyLoss()
        bc_losses = AverageMeter()
        bc_losses_gan = AverageMeter()
        ce_losses = AverageMeter()
        g_losses = AverageMeter()

        for i in range(1, num_steps + 1):
            if i % 10 == 0:
                bc_losses = AverageMeter()
                bc_losses_gan = AverageMeter()
                ce_losses = AverageMeter()
                g_losses = AverageMeter()
                
            scheduler_G.step()

            #Sample Random Noise
            z = torch.randn((100, nz)).cuda()
            B = 50

            labels_l = torch.randint(low=0, high=num_class, size = [B, ]).cuda()
            labels_r = copy.deepcopy(labels_l).cuda()
            labels = torch.stack([labels_l, labels_r]).view(-1)

            '''Train Generator'''
            optimizer_G.zero_grad()
            
            #Get fake image from generator
            fake = generator(z, labels) # pre_x returns the output of G before applying the activation
            
            if i % 10 == 0:
                imgGen = fake.clone()
                imgGen = denormalize(imgGen, target_dataset_name)
                if not os.path.isdir(train_output_path + "/train"):
                    os.mkdir(train_output_path + "/train")
                torchvision.utils.save_image(imgGen, train_output_path + '/train/out_{}.jpg'.format(i * 100 + 100))
            
            # with torch.no_grad(): 
            output = target_model.local_list[0](fake)

            output = target_model.cloud(output)
            
            # Diversity-aware regularization https://sites.google.com/view/iclr19-dsgan/
            
            g_noise_out_dist = torch.mean(torch.abs(fake[:B, :] - fake[B:, :]))
            g_noise_z_dist = torch.mean(torch.abs(z[:B, :] - z[B:, :]).view(B,-1),dim=1)
            g_noise = torch.mean( g_noise_out_dist / g_noise_z_dist ) * D_w


            #Cross Entropy Loss
            ce_loss = criterion(output, labels)

            loss = ce_loss - g_noise
            
            loss.backward()

            optimizer_G.step()

            ce_losses.update(ce_loss.item(), 100)
            
            g_losses.update(g_noise.item(), 100)

            # Log Results
            if i % 10 == 0:
                print(f'Train step: {i}\t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')
        
        logger.debug(f'End of Generator Training: \t CE_Loss: {ce_losses.avg:.10f} diversity_Loss: {g_losses.avg:.10f} bc_losses (G): {bc_losses_gan.avg: .10f}  bc_losses (D)): {bc_losses.avg: .10f}')

        generator.cuda()
        generator.eval()

        z = torch.randn((10, nz)).cuda()
        for i in range(num_class):
            labels = i * torch.ones([10, ]).long().cuda()
            #Get fake image from generator
            fake = generator(z, labels) # pre_x returns the output of G before applying the activation

            imgGen = fake.clone()
            imgGen = denormalize(imgGen, target_dataset_name)
            if not os.path.isdir(train_output_path + "/{}".format(num_steps)):
                os.mkdir(train_output_path + "/{}".format(num_steps))
            torchvision.utils.save_image(imgGen, train_output_path + '/{}/out_{}.jpg'.format(num_steps,"final_label{}".format(i)))

def steal_test(arch, target_model, surrogate_model, val_loader):
    """
    Run evaluation
    """
    # batch_time = AverageMeter()
    fidel_score = AverageMeter()
    top1 = AverageMeter()

    if arch == "ViT":
        dl_transforms = torch.nn.Sequential(
            transforms.Resize(224),
            transforms.CenterCrop(224))
    else:
        dl_transforms = None

    target_model.local_list[0].cuda()
    target_model.local_list[0].eval()
    target_model.cloud.cuda()
    target_model.cloud.eval()
    surrogate_model.local.cuda()
    surrogate_model.local.eval()
    surrogate_model.cloud.cuda()
    surrogate_model.cloud.eval()
    
    for i, (input, target) in enumerate(val_loader):
        if dl_transforms is not None:
            input = dl_transforms(input)
        input = input.cuda()
        target = target.cuda()
        # compute output
        with torch.no_grad():

            output = surrogate_model(input)
            output_target = target_model(input)

        output = output.float()
        output_target = output_target.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        fidel = fidelity(output.data, output_target.data)[0]
        top1.update(prec1.item(), input.size(0))
        fidel_score.update(fidel.item(), input.size(0))
    return top1.avg, fidel_score.avg


def adversarial_attack(logger, target_dataset_name, target_model, surrogate_model, num_class):
    total_succ = 0
    image_shape = get_image_shape(target_dataset_name)
    surrogate_model.local.eval()
    surrogate_model.cloud.eval()
    criterion = nn.CrossEntropyLoss()
    
    '''We run three attacks and select the average'''
    
    '''PGD_target'''
    e = 0.002
    for temporal in range(num_class):
        org_target = temporal
        attack_target = (org_target + 5)%9
        # history_list.append(org_target)
        logger.debug(f"###Round {temporal} src_label: {org_target} target_label: {attack_target}")
        
        
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
            org_image = torch.FloatTensor(1, image_shape[0], image_shape[1], image_shape[2])
            org_image[0] = image_arr[i]
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            # act = target_model.local_list[attack_client](org_image)
            # output = target_model.cloud(act)
            output = target_model(org_image)
            # print(output)
            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]
            fake_image = org_image.clone()
            #modify the original image
            max_val = torch.max(org_image).item()
            min_val = torch.min(org_image).item()
            for iter in range(50):
                # calculate gradient
                grad = torch.zeros(1, image_shape[0], image_shape[1], image_shape[2]).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)
                if fake_image.grad is not None:
                    fake_image.grad.zero_()
                # act = surrogate_model.local(fake_image)
                # output = surrogate_model.cloud(act)
                output = surrogate_model(fake_image)
                loss = criterion(output, fake_label)
                loss.backward()
                #print(loss)
                grad += torch.sign(fake_image.grad)

                fake_image = fake_image - grad * e
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                # act = target_model.local_list[attack_client](fake_image)
                # output = target_model.cloud(act)
                output = target_model(fake_image)
                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred[0, 0]
                
                if fake_label.item() == fake_pred.item() or iter == 49:
                    
                    attack_pred_list = []
                    # act = surrogate_model.local(fake_image)
                    # output = surrogate_model.cloud(act)
                    output = surrogate_model(fake_image)
                    _, attack_pred = output.topk(1, 1, True, True)
                    attack_pred_list.append(attack_pred.data[0, 0].item())

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
        logger.debug(str_log)
        total_succ += succ
    asr_0 = total_succ/image_arr.shape[0]/num_class
    logger.debug(f"PGD_target attack ASR: {asr_0}.")
    

    total_succ = 0
    '''FGSM attack'''
    e = 0.1
    for temporal in range(num_class):
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
            org_image = torch.FloatTensor(1, image_shape[0], image_shape[1], image_shape[2])
            org_image[0] = image_arr[i]
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            # act = target_model.local_list[attack_client](org_image)
            # output = target_model.cloud(act)
            output = target_model(org_image)
            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]

            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]
            
            fake_image = org_image.clone()

            #modify the original image
            max_val = torch.max(org_image).item()
            min_val = torch.min(org_image).item()
            
            # calculate gradient
            grad = torch.zeros(1, image_shape[0], image_shape[1], image_shape[2]).cuda()
            fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

            if fake_image.grad is not None:
                fake_image.grad.zero_()
            # act = surrogate_model.local(fake_image)
            # output = surrogate_model.cloud(act)
            output = surrogate_model(fake_image)
            loss = criterion(output, fake_label)
            loss.backward()
            grad -= torch.sign(fake_image.grad)

            fake_image = fake_image - grad * e # e is epsilon in FGSM:  https://arxiv.org/pdf/1706.06083.pdf
            fake_image[fake_image > max_val] = max_val
            fake_image[fake_image < min_val] = min_val
            # act = target_model.local_list[attack_client](fake_image)
            # output = target_model.cloud(act)
            output = target_model(fake_image)

            _, fake_pred = output.topk(1, 1, True, True)
            fake_pred = fake_pred.data[0, 0]

            attack_pred_list = []
            # act = surrogate_model.local(fake_image)
            # output = surrogate_model.cloud(act)
            output = surrogate_model(fake_image)
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
        logger.debug(str_log)
        total_succ += succ
    asr_1 = total_succ/image_arr.shape[0]/num_class
    logger.debug(f"FGSM attack ASR: {asr_1}.")
    
    '''PGD attack'''
    total_succ = 0
    e = 0.001
    for temporal in range(num_class):
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
            org_image = torch.FloatTensor(1, image_shape[0], image_shape[1], image_shape[2])
            org_image[0] = image_arr[i]
            # print(image_arr[i])
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            # act = surrogate_model.local(org_image)
            # output = surrogate_model.cloud(act)
            output = surrogate_model(org_image)
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
                grad = torch.zeros(1, image_shape[0], image_shape[1], image_shape[2]).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)

                if fake_image.grad is not None:
                    fake_image.grad.zero_()
                # act = surrogate_model.local(fake_image)
                # output = surrogate_model.cloud(act)
                output = surrogate_model(fake_image)
                loss = criterion(output, fake_label)
                loss.backward()
                #print(loss)
                grad -= torch.sign(fake_image.grad)

                fake_image = fake_image - grad * e # e is alpha in PGD:  https://arxiv.org/pdf/1706.06083.pdf
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                # act = surrogate_model.local(fake_image)
                # output = surrogate_model.cloud(act)
                output = surrogate_model(fake_image)
                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred.data[0, 0]

                if fake_label.item() != fake_pred.item() or iter == 49:
                    # print(fake_pred.item(), fake_label.item())
                    attack_pred_list = []

                    # act = surrogate_model.local(fake_image)
                    # output = surrogate_model.cloud(act)
                    output = surrogate_model(fake_image)
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
        logger.debug(str_log)
        total_succ += succ
    asr_2 = total_succ/image_arr.shape[0]/num_class
    logger.debug(f"PGD attack ASR: {asr_2}.")


    total_avg_asr = (asr_0 + asr_1 + asr_2) / 3.
    logger.debug(f"Average attack ASR: {total_avg_asr}.")


    return total_avg_asr

def prepare_steal_attack(logger, save_dir, arch, target_dataset_name,  target_model, attack_style, aux_dataset_name = "cifar10", num_query = 10, num_class = 10, attack_client = 0, data_proportion = 0.2, noniid_ratio = 1.0, last_n_batch = 10000):

    '''last_n_batch: if in resume mode - instead of using all collected input-label pairs, use only the lastest n batch'''
    '''num_query: only make sense in offline, all ME attacks except for Train-ME. SoftTrain-ME and Train-ME is more sensitive on data proportion'''


    # query getting data
    if data_proportion == 0.0:
        attacker_loader_list = [None]
    else:
        attacker_loader_list, _, _ = get_dataset(target_dataset_name, 100, actual_num_users=int(1/data_proportion), noniid_ratio=noniid_ratio)

    attacker_dataloader = attacker_loader_list[attack_client]
    
    ''' Knockoffset, option_B has no prediction query (use grad-matching), option_C has predicion query (craft input-label pair)'''
    
    if "GM_option" in attack_style or "Copycat_option" in attack_style or "Knockoff_option" in attack_style:
        # We fix query batch size to 100 to better control the total query number.
        if data_proportion == 0.0:
            aux_loader_list = [None]
        else:
            aux_loader_list, _, _ = get_dataset(aux_dataset_name, batch_size=100, actual_num_users=int(1/data_proportion))
        aux_loader = aux_loader_list[0]

    # data augmentation
    image_shape = get_image_shape(target_dataset_name)
    
    atk_transforms = torch.nn.Sequential(
        transforms.RandomCrop(image_shape[-1], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    )

    #loss function - knowing exact by default
    criterion = nn.CrossEntropyLoss()

    # prepare model, please make sure model.cloud and classifier are loaded from checkpoint.
    target_model.local_list[attack_client].cuda()
    target_model.local_list[attack_client].eval()
    target_model.cloud.cuda()
    target_model.cloud.eval()

    save_images = []
    save_grad = []
    save_label = []
    save_act = []

    if attack_style == "Craft_option_resume":
        ''' crafting training dataset for surrogate model training'''
        # loading crafted images
        saved_crafted_image_path = save_dir + "/craft_pairs/"
        if os.path.isdir(saved_crafted_image_path):
            print("load from crafted during training")
        else:
            print("No saved pairs is available!")
            exit()

        max_image_id = 0
        for file in glob.glob(saved_crafted_image_path + "*"):
            if "image" in file:
                image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                if image_id > max_image_id:
                    max_image_id = image_id

        for file in glob.glob(saved_crafted_image_path + "*"):
            if "image" in file:
                fake_image = torch.load(file)
                image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                if image_id > max_image_id - last_n_batch: # collect only the last two valid data batch. (even this is very bad)
                    fake_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")
                    fake_grad = torch.load(saved_crafted_image_path + f"grad_{image_id}.pt")

                    save_images.append(fake_image.clone())
                    save_grad.append(fake_grad.clone())
                    save_label.append(fake_label.clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "Craft_option":
        
        craft_LR = 1e-1
        lambda_TV = 0.0
        lambda_l2 = 0.0
        num_craft_step = 50
        num_image_per_class = num_query // num_craft_step // num_class
        
        if num_image_per_class == 0:
            num_craft_step = num_query // num_class
            num_image_per_class = 1
            if num_craft_step == 0:
                print("number of query is tooo small, not going to work.")

        for c in range(num_class):
            fake_label = c * torch.ones((1,)).long().cuda().view(1,)
            
            for i in range(num_image_per_class):

                fake_image = torch.rand([1] + image_shape, requires_grad=True, device="cuda")
                craft_optimizer = torch.optim.Adam(params=[fake_image], lr=craft_LR, amsgrad=True, eps=1e-3) # craft_LR = 1e-1 by default
                for step in range(1, num_craft_step + 1):
                    craft_optimizer.zero_grad()

                    z_private = target_model.local_list[attack_client](fake_image)  # Simulate Original
                    z_private.retain_grad()
                    output = target_model.cloud(z_private)

                    featureLoss = criterion(output, fake_label)

                    TVLoss = lambda_TV * TV(fake_image)
                    normLoss = lambda_l2 * l2loss(fake_image)

                    totalLoss = featureLoss + TVLoss + normLoss

                    totalLoss.backward()

                    craft_optimizer.step()
                    if step == 0 or step == num_craft_step:
                        print("Image {} class {} - Iter {} Feature loss: {} TVLoss: {} l2Loss: {}".format(i, c, step,
                                                                                            featureLoss.cpu().detach().numpy(),
                                                                                            TVLoss.cpu().detach().numpy(),
                                                                                            normLoss.cpu().detach().numpy()))
                
                save_images.append(fake_image.detach().cpu().clone())
                save_grad.append(z_private.grad.detach().cpu().clone())
                save_label.append(fake_label.cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "TrainME_option":
        ''' TrainME: Use available training dataset for surrogate model training'''
        ''' Because of data augmentation, set query to 10 there'''
        if arch == "ViT":
            dl_transforms = torch.nn.Sequential(
                transforms.Resize(224),
                transforms.CenterCrop(224))
        else:
            dl_transforms = None
        for images, labels in attacker_dataloader:
            if dl_transforms is not None:
                images = dl_transforms(images)


            # we will augment during the training.
            images = images.cuda()
            labels = labels.cuda()
            output = target_model(images)

            loss = criterion(output, labels)
            loss.backward(retain_graph = True)
            
            save_images.append(images.cpu().clone())
            save_grad.append(images.cpu().clone())
            save_label.append(labels.cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "SoftTrain_option":

        soft_alpha = 0.9
        ''' SoftTrainME, crafts soft input label pairs for surrogate model training'''
        similar_func = torch.nn.CosineSimilarity(dim = 1)
        
        if num_query < num_class * 100:
            raise("Query budget is too low to run SoftTrainME")
        
        cumulated_query = 0
        while cumulated_query <= num_query: # in query budget, craft soft label 
            
            for i, (images, labels) in enumerate(attacker_dataloader):
                images = atk_transforms(images)
                images = images.cuda()
                labels = labels.cuda()
                
                z_private = target_model.local_list[attack_client](images)
                z_private.retain_grad()
                output = target_model.cloud(z_private)
                loss = criterion(output, labels)
                loss.backward(retain_graph = True)
                z_private_grad = z_private.grad.detach().clone()
                
                cumulated_query += num_class * 100
                if cumulated_query > num_query: # in query budget, craft soft label 
                    break
                
                cos_sim_list = []
                for c in range(num_class):
                    fake_label = c * torch.ones_like(labels).cuda()
                    z_private.grad.zero_()
                    z_private = target_model.local_list[attack_client](images)
                    z_private.retain_grad()
                    output = target_model.cloud(z_private)

                    loss = criterion(output, fake_label)
                    
                    loss.backward(retain_graph = True)
                    fake_z_private_grad = z_private.grad.detach().clone()
                    cos_sim_val = similar_func(fake_z_private_grad.view(labels.size(0), -1), z_private_grad.view(labels.size(0), -1))
                    cos_sim_list.append(cos_sim_val.detach().clone()) # 10 item of [128, 1]
                cos_sim_tensor = torch.stack(cos_sim_list).view(num_class, -1).t().cuda()
                cos_sim_tensor += 1
                cos_sim_sum = (cos_sim_tensor).sum(1) - 1
                derived_label = (1 - soft_alpha) * cos_sim_tensor / cos_sim_sum.view(-1, 1) # [128, 10]

                labels_as_idx = labels.detach().view(-1, 1)
                replace_val = soft_alpha * torch.ones(labels_as_idx.size(), dtype=torch.long).cuda()
                derived_label.scatter_(1, labels_as_idx, replace_val)

                save_images.append(images.cpu().clone())
                save_grad.append(z_private_grad.cpu().clone())
                save_label.append(derived_label.cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "SoftTrain_option_resume":
        # Use SoftTrain_option, query the gradients on inputs with all label combinations
        # UPDATE:  we ignore last n batch here, we move this to training grads.
        soft_alpha = 0.9
        similar_func = torch.nn.CosineSimilarity(dim = 1)
        
        saved_crafted_image_path = save_dir + "/saved_grads/"
        
        if os.path.isdir(saved_crafted_image_path):
            print("load from data collected during training")
        else:
            raise("No saved data is presented!")
            
        max_image_id = 0
        for file in glob.glob(saved_crafted_image_path + "*"):
            if "image" in file and "grad_image" not in file:
                image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                if image_id > max_image_id:
                    max_image_id = image_id

        for image_id in range(max_image_id + 1):
            saved_image = torch.load(saved_crafted_image_path + f"image_{image_id}.pt")
            true_grad = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label0.pt").cuda()
            true_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt").cuda()
            cos_sim_list = []

            for c in range(num_class):
                fake_grad = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{c}.pt").cuda()
                cos_sim_val = similar_func(fake_grad.view(true_label.size(0), -1), true_grad.view(true_label.size(0), -1))
                cos_sim_list.append(cos_sim_val.detach().clone()) # 10 item of [128, 1]

            cos_sim_tensor = torch.stack(cos_sim_list).view(num_class, -1).t().cuda() # [128, 10]
            cos_sim_tensor += 1
            cos_sim_sum = (cos_sim_tensor).sum(1) - 1
            derived_label = (1 - soft_alpha) * cos_sim_tensor / cos_sim_sum.view(-1, 1) # [128, 10]

            
            labels_as_idx = true_label.detach().view(-1, 1)
            replace_val = soft_alpha * torch.ones(labels_as_idx.size(), dtype=torch.long).cuda()
            derived_label.scatter_(1, labels_as_idx, replace_val)

            save_images.append(saved_image.clone())
            save_grad.append(true_grad.detach().cpu().clone())
            save_label.append(derived_label.detach().cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "GM_option":
        target_model.local_list[attack_client].eval()
        target_model.cloud.eval()
        for i, (inputs, target) in enumerate(aux_loader):
            if i * num_class * 100 >= num_query: # limit grad query budget
                break
            
            inputs = atk_transforms(inputs)
            inputs = inputs.cuda()
            
            for j in range(num_class):
                label = j * torch.ones_like(target).cuda()
                
                z_private = target_model.local_list[attack_client](inputs)
                z_private.grad = None
                z_private.retain_grad()
                output = target_model.cloud(z_private)
                
                loss = criterion(output, label)
                loss.backward(retain_graph = True)
                
                z_private_grad = z_private.grad.detach().cpu()
                
                save_grad.append(z_private_grad.clone()) # add a random existing grad.
                save_images.append(inputs.cpu().clone())
                save_label.append(label.cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "GM_option_resume":
        saved_crafted_image_path = save_dir + "/saved_grads/"
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


                image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                
                # saved_image = torch.load(saved_crafted_image_path + f"image_{image_id}.pt")

                if image_id <= max_image_id - last_n_batch: # collect only the last two valid data batch. (even this is very bad)
                    continue
                
                try:
                    for label in range(num_class): # put same-class labels together
                        saved_act = torch.load(saved_crafted_image_path + f"act_{image_id}_label{label}.pt")
                        
                        saved_grads = torch.load(saved_crafted_image_path + f"grad_image{image_id}_label{label}.pt")
                        saved_label = label * torch.ones(saved_grads.size(0), ).long()

                        save_act.append(saved_act.clone())
                        # save_images.append(saved_image.clone())
                        save_grad.append(saved_grads.clone())
                        save_label.append(saved_label.clone())
                except:
                    break
        return save_act, save_grad, save_label
        # return save_images, save_grad, save_label
    
    elif attack_style == "Copycat_option":
        # implement transfer set as copycat paper, here we use cifar-10 dataset.
        target_model.local_list[attack_client].eval()
        target_model.cloud.eval()
        for i, (inputs, target) in enumerate(aux_loader):
            if i * 100 >= num_query:  # limit pred query budget
                break
            inputs = inputs.cuda()
            with torch.no_grad():
                z_private = target_model.local_list[attack_client](inputs)
                output = target_model.cloud(z_private)
                _, pred = output.topk(1, 1, True, True)

                save_images.append(inputs.cpu().clone())
                save_grad.append(torch.zeros((inputs.size(0), z_private.size(1), z_private.size(2), z_private.size(3)))) # add a random existing grad.
                save_label.append(pred.view(-1).cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "Knockoff_option":
        # implement transfer set as knockoff paper, where attacker get access to confidence score
        target_model.local_list[attack_client].eval()
        target_model.cloud.eval()
        for i, (inputs, target) in enumerate(aux_loader):
            if i * 100 >= num_query:   # limit pred query budget
                break
            inputs = inputs.cuda()
            with torch.no_grad():
                z_private = target_model.local_list[attack_client](inputs)
                output = target_model.cloud(z_private)
                softmax_layer = torch.nn.Softmax(dim=None)
                log_prob = softmax_layer(output)
                save_images.append(inputs.cpu().clone())
                save_grad.append(torch.zeros((inputs.size(0), z_private.size(1), z_private.size(2), z_private.size(3)))) # add a random existing grad.
                save_label.append(log_prob.cpu().clone())
        return save_images, save_grad, save_label
    
    elif attack_style == "Generator_option":
        try:
            nz = int(float(save_dir.split("step")[-1].split("-")[0]) * 512)
        except:
            nz = 512
        print(f"latent vector dim: {nz}")
        generator = architectures.GeneratorC(nz=nz, num_classes = num_class, ngf=128, nc=image_shape[0], img_size=image_shape[-1])
        '''GAN_ME, data-free model extraction, train a conditional GAN, train-time option: use 'gan_train' in regularization_option'''
        train_generator(logger, save_dir, target_model, generator, target_dataset_name, num_class, num_query, nz, resume = False)
        return generator
    
    elif attack_style == "Generator_option_resume":
        print(save_dir)
        try:
            nz = int(float(save_dir.split("step")[-1].split("-")[0]) * 512)
        except:
            nz = 512
        print(f"latent vector dim: {nz}")
        generator = architectures.GeneratorC(nz=nz, num_classes = num_class, ngf=128, nc=image_shape[0], img_size=image_shape[-1])
        # get prototypical data using GAN, training generator consumes grad query.
        train_generator(logger, save_dir, target_model, generator, target_dataset_name, num_class, num_query, nz, resume = True)
        return generator
    
    elif attack_style == "Generator_assist_option_resume": #TODO: add a offline MEA version here
        print(save_dir)
        try:
            nz = int(float(save_dir.split("step")[-1].split("-")[0]) * 512)
        except:
            nz = 512
        print(f"latent vector dim: {nz}")
        generator = architectures.GeneratorC(nz=nz, num_classes = num_class, ngf=128, nc=image_shape[0], img_size=image_shape[-1])
        # get prototypical data using GAN, training generator consumes grad query.
        train_generator(logger, save_dir, target_model, generator, target_dataset_name, num_class, num_query, nz, resume = True, assist = True)
        

        saved_crafted_image_path = save_dir + "/saved_grads/"
        
        if os.path.isdir(saved_crafted_image_path):
            print("load from data collected during training")
        else:
            raise("No saved data is presented!")
            
        max_allowed_image_id = 500
        max_image_id = 0
        for file in glob.glob(saved_crafted_image_path + "*"):
            if "image" in file and "grad_image" not in file:
                image_id = int(file.split('/')[-1].split('_')[-1].replace(".pt", ""))
                if image_id > max_image_id:
                    max_image_id = image_id

        for image_id in range(1, min(max_allowed_image_id, max_image_id) + 1):
            saved_image = torch.load(saved_crafted_image_path + f"image_{image_id}.pt")
            true_label = torch.load(saved_crafted_image_path + f"label_{image_id}.pt")

            save_images.append(saved_image.clone())
            save_label.append(true_label.clone())

            del saved_image, true_label

        return generator, save_images, save_label






    else:
        raise("No such MEA attack style")


def steal_attack(save_dir, arch, cutting_layer, num_class, target_model, target_dataset_name, val_loader, aux_dataset_name = "cifar100", batch_size = 128, learning_rate = 0.05, num_query = 10, num_epoch = 200, attack_client=0, attack_style = "TrainME_option", 
                data_proportion = 0.2, noniid_ratio = 1.0, train_clas_layer = -1, surrogate_arch = "same", adversairal_attack_option = False, last_n_batch = 10000):
    # attack_style determines which MEA to perfrom:
    # 
    # data proportion/noniid_ratio is only need for offline MEA. determines 
    # train_clas_layer is the number of unknown layers (counting from the end) left to determine.
    
    model_log_file = save_dir + '/steal_attack.log'
    logger = setup_logger('{}_steal_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
    logger.debug(f"=========RUNING MEA ATTACK - num_cloud_layer {train_clas_layer}, style {attack_style}, data_proportion-{data_proportion}, num_query-{num_query}")
    
    # wandb_name = save_dir.split("/")[-1]
    if "resume" in attack_style:
        wandb_name = "online"
        logger.debug(f"online resume attack setting - Using collected info of last {last_n_batch} batches!")
    else:
        wandb_name = "offline"
    wandb.init(
            # set the wandb project where this run will be logged
            project=f"{wandb_name}-SFL-MEA",
            
            # track hyperparameters and run metadata
            config={
            "file_name": save_dir.split("/")[-1],
            "learning_rate": learning_rate,
            "architecture": arch,
            "dataset": target_dataset_name,
            "noniid_ratio": noniid_ratio,
            "data_proportion": data_proportion,
            "epochs": num_epoch,
            "batch_size": batch_size,
            "cutting_layer": cutting_layer,
            "num_cloud_layer": train_clas_layer,
            "attack_style": attack_style,
            "num_query": num_query,
            "surrogate_arch": surrogate_arch,
            "last_n_batch": last_n_batch,
            }
        )
    
    if train_clas_layer < 0:
        surrogate_model = architectures.create_surrogate_model(arch, cutting_layer, num_class, 0, "same")
    else:
        surrogate_model = architectures.create_surrogate_model(arch, cutting_layer, num_class, train_clas_layer, surrogate_arch)

    #split both model in order to copy the known param of target model to the surrogate one.
    target_model.resplit(train_clas_layer)
    if surrogate_arch == "longer":
        train_clas_layer += 1
    
    if surrogate_arch == "shorter":
        train_clas_layer -= 1
        if train_clas_layer == -1:
            print("train class layer is too small for shorter architecture")
            exit()
    surrogate_model.resplit(train_clas_layer)
    
    milestones = sorted([int(step * num_epoch) for step in [0.2, 0.5, 0.8]])
    optimizer_option = "SGD"

    if attack_style == "GM_option_resume":
        optimizer_option = "Adam"
        print(f"Use Adam optimizer with lr = {learning_rate}")

    start_time = time.time()

    surrogate_model.local.apply(init_weights)
    surrogate_model.cloud.apply(init_weights)

    surrogate_params = []

    if train_clas_layer != -1: # load known params to surrogate model
        w_out =  target_model.local_list[attack_client].state_dict()
        surrogate_model.local.load_state_dict(w_out)
        surrogate_model.local.eval()
        surrogate_model.cloud.train()
        surrogate_params += list(surrogate_model.cloud.parameters())
    else: # if it is -1, the entire surrogate model is unknown
        surrogate_model.local.train()
        surrogate_model.cloud.train()
        surrogate_params += list(surrogate_model.local.parameters()) 
        surrogate_params += list(surrogate_model.cloud.parameters())

    if len(surrogate_params) == 0:
        logger.debug("surrogate parameter got nothing, add dummy param to prevent error")
        dummy_param = torch.nn.Parameter(torch.zero(1,1))
        surrogate_params = dummy_param
    else:
        logger.debug("surrogate parameter has {} trainable layers!".format(len(surrogate_params)))
        
    if optimizer_option == "SGD":
        suro_optimizer = torch.optim.SGD(surrogate_params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        suro_optimizer = torch.optim.Adam(surrogate_params, lr=learning_rate, weight_decay=5e-4)

    suro_scheduler = torch.optim.lr_scheduler.MultiStepLR(suro_optimizer, milestones=milestones,
                                                                gamma=0.2)  # learning rate decay

    # prepare model, please make sure model.cloud and classifier are loaded from checkpoint.
    target_model.local_list[attack_client].cuda()
    target_model.local_list[attack_client].eval()
    target_model.cloud.cuda()
    target_model.cloud.eval()
    
    surrogate_model.local.cuda()
    surrogate_model.cloud.cuda()

    criterion = torch.nn.CrossEntropyLoss()

    # prepare activation/grad/label pairs for the MEA
    # if "GM_option_resume" in attack_style:
    #     if_shuffle = False
    # else:
    #     if_shuffle = True
    if_shuffle = True
    
    if "Generator_option" in attack_style:
        generator = prepare_steal_attack(logger, save_dir, arch, target_dataset_name, target_model, attack_style, aux_dataset_name, num_query, num_class, attack_client, data_proportion, noniid_ratio, last_n_batch)
        generator.eval()
        generator.cuda()
        test_output_path = save_dir + "/generator_test"
        if os.path.isdir(test_output_path):
            rmtree(test_output_path)
        os.makedirs(test_output_path)

    elif "Generator_assist_option" in attack_style:
        generator, save_images, save_label = prepare_steal_attack(logger, save_dir, arch, target_dataset_name, target_model, attack_style, aux_dataset_name, num_query, num_class, attack_client, data_proportion, noniid_ratio, last_n_batch)
        generator.eval()
        generator.cuda()
        test_output_path = save_dir + "/generator_test"
        if os.path.isdir(test_output_path):
            rmtree(test_output_path)
        os.makedirs(test_output_path)
        
        
        save_images = torch.cat(save_images)
        # save_grad = torch.cat(save_grad)
        save_label = torch.cat(save_label)
        indices = torch.arange(save_label.shape[0]).long()

        # if "SoftTrain_option_resume" in attack_style:
        #     indices = torch.flip(indices, dims = [0])
        ds = torch.utils.data.TensorDataset(indices, save_images, save_label)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=4, shuffle=if_shuffle
        )

    else:
        save_images, save_grad, save_label = prepare_steal_attack(logger, save_dir, arch, target_dataset_name, target_model, attack_style, aux_dataset_name, num_query, num_class, attack_client, data_proportion, noniid_ratio, last_n_batch)
        save_images = torch.cat(save_images)
        save_grad = torch.cat(save_grad)
        save_label = torch.cat(save_label)
        indices = torch.arange(save_label.shape[0]).long()

        # if "SoftTrain_option_resume" in attack_style:
        #     indices = torch.flip(indices, dims = [0])
        ds = torch.utils.data.TensorDataset(indices, save_images, save_grad, save_label)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, num_workers=4, shuffle=if_shuffle
        )

        # use in GM
        mean_norm = save_grad.norm(dim=-1).mean().detach().item()


    image_shape = get_image_shape(target_dataset_name)
    # set up data augmentation
    if arch == "ViT":
        dl_transforms = torch.nn.Sequential(
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15))
    else:
        dl_transforms = torch.nn.Sequential(
            transforms.RandomCrop(image_shape[-1], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        )
    
    if "Craft_option" in attack_style or "Generator_option" in attack_style or "GM_option" in attack_style:
        dl_transforms = None
    
    if "Generator_option" in attack_style or "Generator_assist_option" in attack_style:
        try:
            nz = int(float(save_dir.split("step")[-1].split("-")[0]) * 512)
        except:
            nz = 512
    # if attack_style == "SoftTrain_option_resume": #TODO: test the usefulness of this.
    #     dl_transforms = None
    

    time_cost = time.time() - start_time
    logger.debug(f"Time cost on preparing the attack: {time_cost}")
    wandb.run.summary["time-cost-prepare-attack"] = time_cost
    start_time = time.time()

    acc_loss_min = 9.9
    val_acc_max = 0.0
    fidelity_max = 0.0
    best_tail_state_dict = None
    val_accu, fidel_score = steal_test(arch, target_model, target_model, val_loader)
    logger.debug("target-model, val_acc: {}, val_fidelity: {}".format(0, val_accu, fidel_score))
    wandb.run.summary["target-model-orig-val-acc"] = val_accu
    wandb.run.summary["target-model-orig-val-fid"] = fidel_score
    
    # Train surrogate model
    for epoch in range(1, num_epoch + 1):
        grad_loss_list = []
        acc_loss_list = []
        acc_list = []
        suro_scheduler.step(epoch)

        # Use grads only for training surrogate
        if attack_style == "GM_option" or attack_style == "GM_option_resume": 
            acc_loss_list.append(0.0)
            acc_list.append(0.0)
            for idx, (index, image, grad, label) in enumerate(dl):
                image = image.cuda()
                grad = grad.cuda()
                label = label.cuda()
                suro_optimizer.zero_grad()

                if attack_style == "GM_option":
                    with torch.no_grad():
                        act = surrogate_model.local(image)
                    act.requires_grad = True
                    output = surrogate_model.cloud(act)
                else: # GM_option_resume
                    # print(image.size())

                    # smashed_size = target_model.get_smashed_data_size(batch_size)
                    # print(smashed_size)
                    act = torch.clone(image).requires_grad_()
                    # flatten act, otherwise this cause error. [some mistake in the saving process]
                    if act.size(2) == 1 and act.size(3) == 1:
                        act = act.view([act.size(0), act.size(1)])
                    # output = surrogate_model.cloud(act + 1e-3 * torch.randn_like(image))
                    # print(surrogate_model.cloud)
                    output = surrogate_model.cloud(act)
                
                ce_loss = criterion(output, label)
                gradient_loss_style = "l2"
                grad_lambda = 1.0
                grad_approx = torch.autograd.grad(ce_loss, act, create_graph = True)[0] #TODO: debug no grad for GM_resume
                

                if gradient_loss_style == "l2":
                    grad_loss = ((grad - grad_approx).norm(dim=1, p =2)).mean() / mean_norm
                elif gradient_loss_style == "l1":
                    grad_loss = ((grad - grad_approx).norm(dim=1, p =1)).mean() / mean_norm
                elif gradient_loss_style == "cosine":
                    grad_loss = torch.mean(1 - F.cosine_similarity(grad_approx, grad, dim=1))
                total_loss = grad_loss * grad_lambda

                total_loss.backward()
                suro_optimizer.step()

                grad_loss_list.append(total_loss.detach().cpu().item())
            
        elif attack_style == "SoftTrain_option" or attack_style == "SoftTrain_option_resume": #perform matching (distable augmentation)
            
            
            # pretrain use the ground-truth data
            for idx, (index, image, grad, label) in enumerate(dl):
                # if dl_transforms is not None:
                image = dl_transforms(image)
                image = image.cuda()
                grad = grad.cuda()
                label = label.cuda()
                suro_optimizer.zero_grad()
                act = surrogate_model.local(image)
                output = surrogate_model.cloud(act)

                _, real_label = label.max(dim = 1)
                ce_loss = criterion(output, real_label)

                total_loss = ce_loss
                total_loss.backward()
                suro_optimizer.step()
                acc_loss_list.append(ce_loss.detach().cpu().item())

                acc = accuracy(output.data, real_label)[0]
                    
                acc_list.append(acc.cpu().item())

            dl = torch.utils.data.DataLoader(
                ds, batch_size=batch_size, num_workers=4, shuffle=False
            )
            # use the grad dataset to finetune
            for idx, (index, image, grad, label) in enumerate(dl):
                # print(index)
                if idx > len(dl) - last_n_batch:
                    # print(index)
                    image = image.cuda()
                    grad = grad.cuda()
                    label = label.cuda()
                    suro_optimizer.zero_grad()
                    output = surrogate_model(image)
                    _, real_label = label.max(dim = 1)
                    soft_lambda = 0.1
                    ce_loss = (1 - soft_lambda) * criterion(output, real_label) + soft_lambda * torch.mean(torch.sum(-label * torch.nn.functional.log_softmax(output, dim=1), dim=1))
                    total_loss = ce_loss
                    total_loss.backward()
                    suro_optimizer.step()
                    acc_loss_list.append(ce_loss.detach().cpu().item())

                    acc = accuracy(output.data, real_label)[0]
                        
                    acc_list.append(acc.cpu().item())

        elif attack_style == "Generator_option" or attack_style == "Generator_option_resume": # dynamically generates input and label using the trained Generator, used only in GAN-ME
            iter_generator_times = 20
            
            for i in range(iter_generator_times):
                z = torch.randn((batch_size, nz)).cuda()
                label = torch.randint(low=0, high=num_class, size = [batch_size, ]).cuda()
                fake_input = generator(z, label)

                #Save images to file
                if epoch == 1:
                    imgGen = fake_input.clone()
                    imgGen = denormalize(imgGen, target_dataset_name)
                    if not os.path.isdir(test_output_path + "/{}".format(epoch)):
                        os.mkdir(test_output_path + "/{}".format(epoch))
                    torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(epoch, i * batch_size + batch_size))
                suro_optimizer.zero_grad()

                output = surrogate_model(fake_input)

                ce_loss = criterion(output, label)

                total_loss = ce_loss
                
                total_loss.backward()

                suro_optimizer.step()
                
                acc_loss_list.append(ce_loss.detach().cpu().item())
                acc = accuracy(output.data, label)[0]
                acc_list.append(acc.cpu().item())
        
        elif attack_style == "Generator_assist_option_resume":
            
            # for idx, (index, image, grad, label) in enumerate(dl):
            #     # if dl_transforms is not None:
            #     image = dl_transforms(image)

            # try:
            #     index, images, labels = next(dl)
            #     if images.size(0) != batch_size:
            #         client_iterator_list[client_id] = iter(dl)
            #         images, labels = next(client_iterator_list[client_id])
            # except StopIteration:
            #     client_iterator_list[client_id] = iter(self.client_dataloader[idxs_users[client_id]])
            #     images, labels = next(client_iterator_list[client_id])
            # iter_generator_times = 20
            
            for idx, (index, image, label) in enumerate(dl):

                image = dl_transforms(image)

                image = image.cuda()

                label = label.cuda()

                z = torch.randn((batch_size, nz)).cuda()
                
                # get images and labels from dl,

                # label = torch.randint(low=0, high=num_class, size = [batch_size, ]).cuda()
                
                random_mask = torch.randint(low=0, high=2, size = [batch_size, ]).cuda()

                noise = generator(z, label)
                # print(noise.size())
                # print(image.size())

                # print(noise)
                # print(image)


                # fake_input = random_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3) * noise   + image
                fake_input = noise   + image

                #TODO: compare with baseline
                # fake_input = image

                # fake_input = random_dropped_noise  + image

                #Save images to file
                if epoch == 1:
                    imgGen = fake_input.clone()
                    imgGen = denormalize(imgGen, target_dataset_name)
                    if not os.path.isdir(test_output_path + "/{}".format(epoch)):
                        os.mkdir(test_output_path + "/{}".format(epoch))
                    torchvision.utils.save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(epoch, idx * batch_size + batch_size))
                suro_optimizer.zero_grad()

                output = surrogate_model(fake_input)

                ce_loss = criterion(output, label)

                total_loss = ce_loss
                
                total_loss.backward()

                suro_optimizer.step()
                
                acc_loss_list.append(ce_loss.detach().cpu().item())
                acc = accuracy(output.data, label)[0]
                acc_list.append(acc.cpu().item())
        
        else: # dl contains input and label, very general framework
             
            for idx, (index, image, grad, label) in enumerate(dl):
                if dl_transforms is not None:
                    image = dl_transforms(image)
                
                image = image.cuda()
                grad = grad.cuda()
                label = label.cuda()

                suro_optimizer.zero_grad()
                # act = surrogate_model.local(image)
                # output = surrogate_model.cloud(act)
                output = surrogate_model(image)

                if "Knockoff_option" in attack_style:
                    log_prob = torch.nn.functional.log_softmax(output, dim=1)
                    ce_loss = torch.mean(torch.sum(-label * log_prob, dim=1))
                else:
                    ce_loss = criterion(output, label)

                total_loss = ce_loss
                total_loss.backward()
                suro_optimizer.step()

                acc_loss_list.append(ce_loss.detach().cpu().item())

                if "Knockoff_option" in attack_style:
                    _, real_label = label.max(dim=1)
                    acc = accuracy(output.data, real_label)[0]
                else:
                    acc = accuracy(output.data, label)[0]
                    
                acc_list.append(acc.cpu().item())
        
        
        if "GM_option" in attack_style:
            grad_loss_mean = np.mean(grad_loss_list)
        else:
            grad_loss_mean = None
        
        acc_loss_mean = np.mean(acc_loss_list)
        avg_acc = np.mean(acc_list)

        val_accu, fidel_score = steal_test(arch, target_model, surrogate_model, val_loader)

        if val_accu > val_acc_max:
            acc_loss_min = acc_loss_mean
            val_acc_max = val_accu
            acc_max_fidelity = fidel_score
            best_client_state_dict = surrogate_model.local.state_dict()
            best_tail_state_dict = surrogate_model.cloud.state_dict()
        if fidel_score > fidelity_max:
            acc_loss_min = acc_loss_mean
            fidelity_max = fidel_score
            fidel_max_acc = val_accu
            closest_client_state_dict = surrogate_model.local.state_dict()
            closest_tail_state_dict = surrogate_model.cloud.state_dict()
        
        logger.debug("epoch: {}, train_acc: {}, val_acc: {}, fidel_score: {}, acc_loss: {}, grad_loss: {}".format(epoch, avg_acc, val_accu, fidel_score, acc_loss_mean, grad_loss_mean))
        wandb.log({"surrogate-model-train-acc": avg_acc, "surrogate-model-val-acc": val_accu, "surrogate-model-val-fid": fidel_score, "surrogate-model-train-loss": acc_loss_mean, "surrogate-model-GM-loss": grad_loss_mean})
    time_cost = time.time() - start_time
    logger.debug(f"Time cost on training surrogate model: {time_cost}")
    wandb.run.summary["time-cost-train-surrogate"] = time_cost
    if best_tail_state_dict is not None:
        logger.debug("load best stealed model.")
        logger.debug("Best perform model, val_acc: {}, fidel_score: {}, acc_loss: {}".format(val_acc_max, acc_max_fidelity, acc_loss_min))
        surrogate_model.local.load_state_dict(best_client_state_dict)
        surrogate_model.cloud.load_state_dict(best_tail_state_dict)
    
    if closest_tail_state_dict is not None:
        logger.debug("load cloest stealed model.")
        logger.debug("Cloest perform model, val_acc: {}, fidel_score: {}, acc_loss: {}".format(fidel_max_acc, fidelity_max, acc_loss_min))
        surrogate_model.local.load_state_dict(closest_client_state_dict)
        surrogate_model.cloud.load_state_dict(closest_tail_state_dict)

    

    if adversairal_attack_option:
        average_ASR = adversarial_attack(logger, target_dataset_name, target_model, surrogate_model, num_class)
        wandb.run.summary["average ASR"] = average_ASR
    wandb.finish()

    