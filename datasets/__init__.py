import sys
import torch
sys.path.append("../")
from datasets.datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_imagenet_trainloader, get_imagenet_testloader, get_mnist_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_femnist_bothloader, get_tinyimagenet_bothloader

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "mnist" or dataset == "fmnist" or dataset == "femnist":
        return torch.clamp((x + 1)/2, 0, 1)
    elif dataset == "cifar10":
        std = [0.247, 0.243, 0.261]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "facescrub":
        std = (0.2058, 0.2275, 0.2098)
        mean = (0.5708, 0.5905, 0.4272)
    elif dataset == "tinyimagenet":
        mean = (0.5141, 0.5775, 0.3985)
        std = (0.2927, 0.2570, 0.1434)
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = tensor[t].mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)

def get_dataset(dataset_name, batch_size = 128, noniid_ratio = 1.0, actual_num_users = 10, augmentation_option = False, last_client_fix_amount = -1):
    if dataset_name == "cifar10":
        client_dataloader, _ , _ = get_cifar10_trainloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=True,
                                                                        num_client=actual_num_users,
                                                                        augmentation_option = augmentation_option,
                                                                        noniid_ratio = noniid_ratio,
                                                                        last_client_fix_amount = last_client_fix_amount)
        pub_dataloader, _ , _ = get_cifar10_testloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=False)
        orig_class = 10
    elif dataset_name == "cifar100":
        client_dataloader, _ , _ = get_cifar100_trainloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=True,
                                                                        num_client=actual_num_users,
                                                                        augmentation_option = augmentation_option,
                                                                        noniid_ratio = noniid_ratio,
                                                                        last_client_fix_amount = last_client_fix_amount)

        pub_dataloader, _ , _ = get_cifar100_testloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=False)
        orig_class = 100

    elif dataset_name == "imagenet":
        client_dataloader = get_imagenet_trainloader(batch_size=batch_size,
                                                            num_workers=4,
                                                            shuffle=True,
                                                            num_client=actual_num_users,
                                                            augmentation_option = augmentation_option,
                                                            noniid_ratio = noniid_ratio,
                                                            last_client_fix_amount = last_client_fix_amount)
        pub_dataloader = get_imagenet_testloader(batch_size=batch_size,
                                                        num_workers=4,
                                                        shuffle=False)
        orig_class = 1000
    elif dataset_name == "svhn":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()

        client_dataloader, _, _ = get_SVHN_trainloader(batch_size=batch_size,
                                                                num_workers=4,
                                                                shuffle=True,
                                                                num_client=actual_num_users,
                                                                augmentation_option = augmentation_option,
                                                                last_client_fix_amount = last_client_fix_amount)
        pub_dataloader, _, _ = get_SVHN_testloader(batch_size=batch_size,
                                                                num_workers=4,
                                                                shuffle=False)
        orig_class = 10

    elif dataset_name == "tinyimagenet":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_tinyimagenet_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            augmentation_option = augmentation_option,
                                                                            last_client_fix_amount = last_client_fix_amount)
        orig_class = 200
    elif dataset_name == "mnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_mnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            last_client_fix_amount = last_client_fix_amount)
        orig_class = 10
    elif dataset_name == "fmnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_fmnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            last_client_fix_amount = last_client_fix_amount)
        orig_class = 10
    elif dataset_name == "femnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_femnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            last_client_fix_amount = last_client_fix_amount)
        orig_class = 62
    else:
        raise ("Dataset {} is not supported!".format(dataset_name))
    
    return client_dataloader, pub_dataloader, orig_class

def get_image_shape(dataset_name):
    if dataset_name == "cifar10":
        image_shape = [3, 32, 32]
    elif dataset_name == "cifar100":
        image_shape = [3, 32, 32]
    elif dataset_name == "imagenet":
        image_shape = [3, 224, 224]
    elif dataset_name == "svhn":
        image_shape = [3, 32, 32]
    elif dataset_name == "tinyimagenet":
        image_shape = [3, 32, 32]
    elif dataset_name == "mnist":
        image_shape = [3, 32, 32]
    elif dataset_name == "fmnist":
        image_shape = [3, 32, 32]
    elif dataset_name == "femnist":
        image_shape = [3, 32, 32]
    else:
        raise ("Dataset {} is not supported!".format(dataset_name))
    
    return image_shape