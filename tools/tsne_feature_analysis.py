
import torch
from sklearn.manifold import TSNE
import os
import numpy as np
import matplotlib.pyplot as plt
model_name = "naive_vgg11_cifar10_cut10" # get model state dict using naive ME
cls = 0

for cls in range(10):
    save_dir = f"./tools/feature_vectors/{model_name}/deep_features/{cls}/"

    tensor_list = []
    name_list = []
    # walk all the files in the current directory
    for filename in os.listdir(save_dir):
        if ".pt" in filename:
            pt = torch.load(save_dir + filename)
            tensor_list.append(pt)
            
            if "real" in filename:
                name_list.append("Real")
            elif "randommix_sameclass" in filename:
                name_list.append("SameClassMix")
            elif "randommix" in filename and "str0.2" in filename:
                name_list.append("UnderMix")
            elif "randommix" in filename and "str0.6" in filename:
                name_list.append("ProperMix")
            else:
                name_list.append("NoMix")

    data = torch.cat(tensor_list).numpy() # [150, 512, 1, 1]

    data = data.reshape(data.shape[0], -1) # [150, 512]

    data = (data - data.mean()) / data.std()

    # Create a t-SNE object
    tsne = TSNE(n_components=2)

    # Apply t-SNE on the data
    tsne_data = tsne.fit_transform(data)

    # Create a figure and subplot
    fig, ax = plt.subplots()
    # Iterate over each tensor and its corresponding t-SNE coordinates
    for i in range(len(tensor_list)):

        x = tsne_data[tensor_list[0].size(0) * i:tensor_list[0].size(0) * (i + 1), 0]  # Get the t-SNE coordinates
        y = tsne_data[tensor_list[0].size(0) * i:tensor_list[0].size(0) * (i + 1), 1]  # Get the t-SNE coordinates

        label_name = name_list[i]    
        # Plot the point with a different color for each tensor
        ax.scatter(x, y, label=label_name, alpha=0.7)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
    plt.savefig(f"{save_dir}tsne.png")