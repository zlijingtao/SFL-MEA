import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_ssim
from utils import accuracy, fidelity, AverageMeter, TV, l2loss, setup_logger, get_PSNR, apply_transform_test, apply_transform
from datasets import get_dataset, denormalize, get_image_shape
from torchvision.utils import save_image, make_grid
from shutil import rmtree
from torch.autograd import Variable
import logging
import models.architectures_torch as architectures
import wandb
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



def gen_ir(aux_dataset_name, val_single_loader, target_model, img_folder="./tmp", intermed_reps_folder="./tmp", select_layer_output=-1):
    """
    At server side, query client-side model to Generate (Raw Input - Intermediate Representation) Pair for Training of the AutoEncoder
    """
    img_folder = os.path.abspath(img_folder)
    intermed_reps_folder = os.path.abspath(intermed_reps_folder)
    if not os.path.isdir(intermed_reps_folder):
        os.makedirs(intermed_reps_folder)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)
    
    # switch to evaluate mode - 
    target_model.eval()
    
    file_id = 0
    for i, (input, target) in enumerate(val_single_loader):
        # input = input.cuda(async=True)
        input = input.cuda()
        target = target.item()

        # compute output
        if select_layer_output == -1: # when client side model is a standalone queryable module
            target_model.local_list[0].eval()
            with torch.no_grad():
                ir = target_model.local_list[0](input)
        else: # when client side model is part of the entire model
            activation_4 = {}
            def get_activation_4(name):
                def hook(model, input, output):
                    activation_4[name] = output.detach()
                return hook
            with torch.no_grad():
                activation_4 = {}
                count = 0
                for name, m in target_model.named_modules():
                    if select_layer_output == count:
                        m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                        valid_key = "ACT-{}".format(name)
                        break
                    count += 1
                output = target_model(ir)
            try:
                ir = activation_4[valid_key]
            except:
                print("cannot attack from later layer, server-side model is empty or does not have enough layers")
        
        ir = ir.float()
        inp_img_path = "{}/{}.jpg".format(img_folder, file_id)
        out_tensor_path = "{}/{}.pt".format(intermed_reps_folder, file_id)
        input = denormalize(input, aux_dataset_name)
        save_image(input, inp_img_path)
        torch.save(ir.cpu(), out_tensor_path)
        file_id += 1
    print("Overall size of Training/Validation Datset for AE is {}: {}".format(int(file_id * 0.9),
                                                                                int(file_id * 0.1)))


def save_image_act_pair(save_dir, target_model, target_dataset_name, input, target, client_id, epoch, select_layer_output=-1):
    """
        Run one train epoch - Generate test set for MIA decoder
    """
    path_dir = os.path.join(save_dir, 'save_activation_client_{}_epoch_{}'.format(client_id, epoch))
    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)
    else:
        rmtree(path_dir)
        os.makedirs(path_dir)
    input = input.cuda()

    for j in range(input.size(0)):
        img = input[None, j, :, :, :]
        label = target[None, j]
        
        if select_layer_output == -1:
            with torch.no_grad():
                target_model.local_list[client_id].eval()
                save_activation = target_model.local_list[client_id](img)
        else:
            target_model.local_list[client_id].eval()
            target_model.cloud.eval()

            activation_3 = {}

            def get_activation_3(name):
                def hook(model, input, output):
                    activation_3[name] = output.detach()

                return hook

            with torch.no_grad():
                activation_3 = {}
                count = 0
                for name, m in target_model.named_modules():
                    if select_layer_output == count:
                        m.register_forward_hook(get_activation_3("ACT-{}".format(name)))
                        valid_key = "ACT-{}".format(name)
                        break
                    count += 1
                output = target_model(img)
            try:
                save_activation = activation_3[valid_key]
            except:
                print("cannot attack from later layer, server-side model is empty or does not have enough layers")
        
        img = denormalize(img, target_dataset_name)
        save_activation = save_activation.float()
        
        save_image(img, os.path.join(path_dir, "{}.jpg".format(j)))
        torch.save(save_activation.cpu(), os.path.join(path_dir, "{}.pt".format(j)))
        torch.save(label.cpu(), os.path.join(path_dir, "{}.label".format(j)))

def pack_images_to_wandb(dir, wandb_key_name):
    image_list = []
    # out_image_list = []
    # for filename in os.listdir(dir):
    for i in range(1, 129):
        # if "inp_" in filename:
        image_list.append(wandb.Image(dir + f"/inp_{int(32*i)}.jpg"))
        image_list.append(wandb.Image(dir + f"/out_{int(32*i)}.jpg"))

    wandb.log({f"{wandb_key_name}": image_list})


def MIA_attack(save_dir, target_model, target_dataset_name = "cifar10", aux_dataset_name = "cifar10", num_epochs = 50, attack_option="MIA", 
                target_client=0, gan_AE_type = "res_normN4C64", loss_type="MSE", select_layer_output=-1, MIA_optimizer = "Adam", MIA_lr = 1e-3):

    # get saved image_act pairs Use a fix set of testing image for each experiment
    wandb_name = save_dir.split("/")[-1]
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{wandb_name}-SFL-MIA",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": MIA_lr,
        "architecture": gan_AE_type,
        "dataset": target_dataset_name,
        "aux_dataset": aux_dataset_name,
        "epochs": num_epochs,
        "MIA_optimizer": MIA_optimizer,
        "select_layer_output": select_layer_output,
        "num_local_layer": target_model.get_num_of_local_layer(),
        "num_cloud_layer": target_model.get_num_of_cloud_layer(),
        "attack_option": attack_option,
        "loss_type": loss_type
        }
    )

    if target_dataset_name == "cifar10":
        images = torch.load("./saved_tensors/test_cifar10_image.pt")
        labels = torch.load("./saved_tensors/test_cifar10_label.pt")
    elif target_dataset_name == "svhn":
        images = torch.load("./saved_tensors/test_svhn_image.pt")
        labels = torch.load("./saved_tensors/test_svhn_label.pt")
    elif target_dataset_name == "cifar100":
        images = torch.load("./saved_tensors/test_cifar100_image.pt")
        labels = torch.load("./saved_tensors/test_cifar100_label.pt")
    elif target_dataset_name == "mnist":
        images = torch.load("./saved_tensors/test_mnist_image.pt")
        labels = torch.load("./saved_tensors/test_mnist_label.pt")
    elif target_dataset_name == "fmnist":
        images = torch.load("./saved_tensors/test_fmnist_image.pt")
        labels = torch.load("./saved_tensors/test_fmnist_label.pt")
    elif target_dataset_name == "facescrub":
        images = torch.load("./saved_tensors/test_facescrub_image.pt")
        labels = torch.load("./saved_tensors/test_facescrub_label.pt")
    else:
        raise(f"MIA on {target_dataset_name} IS Not supported")
    # image_grid = make_grid(images, 8)
    # image_grid = wandb.Image(image_grid, caption="Original Images")
        
    # wandb.log({"original-images": image_grid})


    save_image_act_pair(save_dir, target_model, target_dataset_name, images, labels, target_client, 200, select_layer_output= select_layer_output)

    # setup gan_adv regularizer
    gan_AE_activation = "sigmoid"
    
    attack_option = attack_option
    MIA_optimizer = MIA_optimizer
    MIA_lr = MIA_lr
    attack_batchsize = 32
    attack_num_epochs = num_epochs
    model_log_file = save_dir + '/{}_{}_{}_cut{}.log'.format(attack_option, loss_type, aux_dataset_name, target_model.get_num_of_local_layer())
    logger = setup_logger('{}_attack_logger'.format(str(save_dir)), model_log_file, level=logging.DEBUG)
   
    # pass
    image_data_dir = save_dir + "/img"
    tensor_data_dir = save_dir + "/img"

    # Clear content of image_data_dir/tensor_data_dir
    if os.path.isdir(image_data_dir):
        rmtree(image_data_dir)
    if os.path.isdir(tensor_data_dir):
        rmtree(tensor_data_dir)
    _, val_single_loader, _ = get_dataset(aux_dataset_name, 1, actual_num_users=1)

    attack_path = save_dir + '/{}_{}_{}_cut{}'.format(attack_option, loss_type, aux_dataset_name, target_model.get_num_of_local_layer())
    if not os.path.isdir(attack_path):
        os.makedirs(attack_path)
        os.makedirs(attack_path + "/train")
        os.makedirs(attack_path + "/test")
    train_output_path = "{}/train".format(attack_path)
    test_output_path = "{}/test".format(attack_path)
    model_path = "{}/model.pt".format(attack_path)
    path_dict = {"model_path": model_path, "train_output_path": train_output_path,
                    "test_output_path": test_output_path}

    if attack_option == "MIA": # launch model-based MIA attack
        logger.debug("Generating IR ...... (may take a while)")

        gen_ir(aux_dataset_name, val_single_loader, target_model, image_data_dir, tensor_data_dir,
                    select_layer_output=select_layer_output)

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

        decoder = architectures.get_decoder(gan_AE_type, input_nc = input_nc, output_nc=3, input_dim=input_dim, output_dim=32, gan_AE_activation=gan_AE_activation)
        
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
        sp_testloader = apply_transform_test(1, save_dir + "/save_activation_client_{}_epoch_{}".format(
            target_client, 200), save_dir + "/save_activation_client_{}_epoch_{}".format(target_client,
                                                                                                        200))

        # Perform Input Extraction Attack
        model_based_MIA(attack_num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict,
                    attack_batchsize, loss_type=loss_type)

        
        # malicious_option = True if "local_plus_sampler" in args.MA_fashion else False
        mse_score, ssim_score, psnr_score = test_attack(attack_num_epochs, decoder, sp_testloader, logger,
                                                                path_dict, attack_batchsize)
        # wandb.Table(columns=["MSE", "SSIM", "PSNR"], 
        #     data=[[mse_score, ssim_score, psnr_score]])
        
        wandb.run.summary["MSE"] = mse_score
        wandb.run.summary["SSIM"] = ssim_score
        wandb.run.summary["PSNR"] = psnr_score
        
        pack_images_to_wandb(test_output_path + f"/{attack_num_epochs}", wandb_key_name="orig-reconst-images")

        # Clear content of image_data_dir/tensor_data_dir
        if os.path.isdir(image_data_dir):
            rmtree(image_data_dir)
        if os.path.isdir(tensor_data_dir):
            rmtree(tensor_data_dir)
        wandb.finish()
        return mse_score, ssim_score, psnr_score
    
    elif attack_option == "MIA_mf":  # launch Model-free MIA, does not need a AE model, optimize each fake image instead.
        lambda_TV = 0.0
        lambda_l2 = 0.0
        num_step = attack_num_epochs * 60

        sp_testloader = apply_transform_test(1, save_dir + "/save_activation_client_{}_epoch_{}".format(
            target_client, 200), save_dir + "/save_activation_client_{}_epoch_{}".format(target_client, 200))
        criterion = nn.MSELoss().cuda()
        ssim_loss = pytorch_ssim.SSIM()
        all_test_losses = AverageMeter()
        ssim_test_losses = AverageMeter()
        psnr_test_losses = AverageMeter()

        target_model.local_list[0].eval()
        
        for num, data in enumerate(sp_testloader):
            
            img, ir, _ = data
            # optimize a fake_image to (1) have similar ir, (2) have small total variance, (3) have small l2
            img = img.cuda()
            ir = ir.cuda()
            
            fake_image = torch.zeros(img.size(), requires_grad=True, device="cuda")
            nn.init.uniform_(fake_image)
            # optimizer = torch.optim.Adam(params=[fake_image], lr=8e-1, amsgrad=True, eps=1e-3)
            optimizer = torch.optim.Adam(params = [fake_image], lr = MIA_lr, amsgrad=True, eps=1e-3)
            for step in range(1, num_step + 1):
                optimizer.zero_grad()
                
                if select_layer_output == -1:
                    fake_ir = target_model.local_list[0](fake_image)  # Simulate Original
                else:

                    activation_4 = {}
                    def get_activation_4(name):
                        def hook(model, input, output):
                            activation_4[name] = output.detach()
                        return hook
                    count = 0
                    for name, m in target_model.named_modules():
                        if select_layer_output == count:
                            m.register_forward_hook(get_activation_4("ACT-{}".format(name)))
                            valid_key = "ACT-{}".format(name)
                            break
                        count += 1
                    output = target_model(ir)
                    try:
                        fake_ir = activation_4[valid_key]
                    except:
                        print("cannot attack from later layer, server-side model is empty or does not have enough layers")
                
                featureLoss = criterion(fake_ir, ir)
                TVLoss = TV(fake_image)
                normLoss = l2loss(fake_image)
                totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss
                totalLoss.backward()
                optimizer.step()
                

                if step == 0 or step == num_step:
                    wandb.log({"train-loss": featureLoss.cpu().detach().numpy()})
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
            # imgGen = denormalize(imgGen, aux_dataset_name)
            save_image(imgGen, test_output_path + '/{}/out_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
            # imgOrig = denormalize(imgOrig, aux_dataset_name)
            save_image(imgOrig, test_output_path + '/{}/inp_{}.jpg'.format(attack_num_epochs,
                                                                                                num * attack_batchsize + attack_batchsize))
        pack_images_to_wandb(test_output_path + f"/{attack_num_epochs}", wandb_key_name="orig-reconst-images")
        
        logger.debug("MSE Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            all_test_losses.avg))
        logger.debug("SSIM Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            ssim_test_losses.avg))
        logger.debug("PSNR Loss on ALL Image is {:.4f} (Real Attack Results on the Target Client)".format(
            psnr_test_losses.avg))
        
        wandb.run.summary["MSE"] = all_test_losses.avg
        wandb.run.summary["SSIM"] = ssim_test_losses.avg
        wandb.run.summary["PSNR"] = psnr_test_losses.avg
        wandb.finish()
        return all_test_losses.avg, ssim_test_losses.avg, psnr_test_losses.avg

def model_based_MIA(num_epochs, decoder, optimizer, trainloader, testloader, logger, path_dict, batch_size, loss_type="MSE"):
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
        raise ("No such loss in loss_type")
    device = next(decoder.parameters()).device
    decoder.train()
    for epoch in range(round_ * num_epochs, (round_ + 1) * num_epochs):
        for num, data in enumerate(trainloader, 1):
            img, ir = data
            img, ir = img.type(torch.FloatTensor), ir.type(torch.FloatTensor)
            img, ir = Variable(img).to(device), Variable(ir).to(device)

            # print(ir.size())
            output = decoder(ir)

            if loss_type == "MSE":
                reconstruction_loss = criterion(output, img)
            elif loss_type == "SSIM":
                reconstruction_loss = -criterion(output, img)
            elif loss_type == "PSNR":
                reconstruction_loss = -1 / 10 * get_PSNR(img, output)
            else:
                raise ("No such loss in loss_type")
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

        # torch.save(decoder.state_dict(), path_dict["model_path"])
        logger.debug(
            "epoch [{}/{}], train_loss {train_losses.val:.4f} ({train_losses.avg:.4f}), val_loss {val_losses.val:.4f} ({val_losses.avg:.4f})".format(
                epoch + 1,
                num_epochs, train_losses=train_losses, val_losses=val_losses))
        wandb.log({"train-loss": train_losses.avg, "val-loss": val_losses.avg})
    
    if loss_type == "MSE":
        logger.debug("Best Validation Loss is {}".format(min_val_loss))
        best_val_loss = min_val_loss
        wandb.run.summary["best-val-loss"] = best_val_loss
        return best_val_loss
    elif loss_type == "SSIM":
        logger.debug("Best Validation Loss is {}".format(max_val_loss))
        best_val_loss = max_val_loss
        wandb.run.summary["best-val-loss"] = best_val_loss
        return best_val_loss
    elif loss_type == "PSNR":
        logger.debug("Best Validation Loss is {}".format(max_val_loss))
        best_val_loss = max_val_loss
        wandb.run.summary["best-val-loss"] = best_val_loss
        return best_val_loss
    

def test_attack(num_epochs, decoder, sp_testloader, logger, path_dict, batch_size):
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

