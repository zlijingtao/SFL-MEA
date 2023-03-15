
# %%
import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np  
import torch
import torch.distributions as distributions  
from bhtsne import bhtsne 
import pandas as pd

# %%
'''Get bhtSNE plot for activation data.'''
folder_name = "saves/baseline"
file_name = "ace_V2_epoch_vgg11_bn_cutlayer_4_client_1_seed125_dataset_cifar10_lr_0.05_200epoch"
# tensor_file_name = "TrainME_option_ACT-9_act.txt"
# tensor_file_name = "TrainME_option_ACT-20_act.txt"
N = 8
if N == 2:
    key_name = "ACT-20"
elif N == 5:
    key_name = "ACT-9"
elif N == 8:
    key_name = "z_private"
tensor_file_name = f"TrainME_option_{key_name}_act.txt"
tensor_label_name = f"TrainME_option_{key_name}_target.txt"
tensor_path = "./" + folder_name + "/"+ file_name + "/" + tensor_file_name
label_path = "./" + folder_name + "/"+ file_name + "/" + tensor_label_name

data = np.loadtxt(tensor_path)
label = np.loadtxt(label_path)

embedding_array = bhtsne.run_bh_tsne(data, initial_dims=data.shape[1])

print(embedding_array)

# %%
palette=["#1CE6FF", "#FF34FF", "#FF4A46","#008941", "#006FA6", "#A30059", '#008080','#FFA500', '#3399FF','#800080']

for i in range(10):
    pos=(label==i).flatten()
    plt.scatter(embedding_array[pos,0], embedding_array[pos,1],
        c=palette[i],
        label=str(i),
        edgecolor='black'
    )

plt.title(f't-SNE on intermediate activation (N = {N})', fontsize=18)
plt.xlabel('t-SNE 1', fontsize=18)
plt.ylabel('t-SNE 2', fontsize=18)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
print("Saving plot")
plt.savefig(f'result_N_{N}.pdf')
print('Showing plot, voila !')
plt.show()

# %%
