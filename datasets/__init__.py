import sys
sys.path.append("../")
from datasets.datasets_torch import get_cifar100_trainloader, get_cifar100_testloader, get_cifar10_trainloader, \
    get_cifar10_testloader, get_imagenet_trainloader, get_imagenet_testloader, get_mnist_bothloader, get_SVHN_trainloader, get_SVHN_testloader, get_fmnist_bothloader, get_femnist_bothloader, get_tinyimagenet_bothloader


def get_dataset(dataset_name, batch_size = 128, noniid_ratio = 1.0, actual_num_users = 10, collude_use_public = False, augmentation_option = False):
    if dataset_name == "cifar10":
        client_dataloader, _ , _ = get_cifar10_trainloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=True,
                                                                        num_client=actual_num_users,
                                                                        collude_use_public=collude_use_public,
                                                                        augmentation_option = augmentation_option,
                                                                        noniid_ratio = noniid_ratio)
        pub_dataloader, _ , _ = get_cifar10_testloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=False)
        orig_class = 10
    elif dataset_name == "cifar100":
        client_dataloader, _ , _ = get_cifar100_trainloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=True,
                                                                        num_client=actual_num_users,
                                                                        collude_use_public=collude_use_public,
                                                                        augmentation_option = augmentation_option,
                                                                        noniid_ratio = noniid_ratio)

        pub_dataloader, _ , _ = get_cifar100_testloader(batch_size=batch_size,
                                                                        num_workers=4,
                                                                        shuffle=False)
        orig_class = 100

    elif dataset_name == "imagenet":
        client_dataloader = get_imagenet_trainloader(batch_size=batch_size,
                                                            num_workers=4,
                                                            shuffle=True,
                                                            num_client=actual_num_users,
                                                            collude_use_public=collude_use_public,
                                                            augmentation_option = augmentation_option,
                                                            noniid_ratio = noniid_ratio)
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
                                                                collude_use_public=collude_use_public)
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
                                                                            collude_use_public=collude_use_public)
        orig_class = 200
    elif dataset_name == "mnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_mnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            collude_use_public=collude_use_public)
        orig_class = 10
    elif dataset_name == "fmnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_fmnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            collude_use_public=collude_use_public)
        orig_class = 10
    elif dataset_name == "femnist":
        if noniid_ratio != 1.0:
            print("Non IID in current dataset is not supported!")
            exit()
        client_dataloader, pub_dataloader = get_femnist_bothloader(batch_size=batch_size, 
                                                                            num_workers=4,
                                                                            shuffle=True,
                                                                            num_client=actual_num_users,
                                                                            collude_use_public=collude_use_public)
        orig_class = 62
    else:
        raise ("Dataset {} is not supported!".format(dataset_name))
    
    return client_dataloader, pub_dataloader, orig_class