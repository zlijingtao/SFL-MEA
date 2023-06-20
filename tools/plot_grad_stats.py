import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':
    folder_name = "./saves/V1-grad-stats"
    file_name = "V1-vgg11-cifar10-naive_train_ME_surrogate_print_grad_stats_start0-str1.0-cut10-client5-noniid1.0--data50"
    
    mean_list = []
    std_list = []

    total_num_epoch = 200

    for epoch in range(1, total_num_epoch+1):
        grad_path = folder_name + "/"+ file_name + "/grad_stats/" + f"grad_epoch_{epoch}_batch0.pt"
        grad = torch.load(grad_path)
        mean_list.append(torch.mean(grad).item())
        std_list.append(torch.std(grad).item())
    
    mean_array = np.array(mean_list)
    std_array = np.array(std_list)

    np.savetxt(f"{folder_name}/{file_name}/grad_stats/grad_mean.txt", mean_array, delimiter=",")
    np.savetxt(f"{folder_name}/{file_name}/grad_stats/grad_std.txt", std_array, delimiter=",")