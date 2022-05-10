import glob
import os
# search_path = "./ideal_extraction_cifar10_correct"
# target_layer = 8
# for subdir, dirs, files in os.walk(search_path):
#     if "stealtype" in subdir and "layer_-1" not in subdir:
#         # print(subdir)
#         layer = int(float(subdir.split('layer_')[1].split('_data')[0]))
#         exp_name = subdir.split('stealtype_')[1].split('_epoch300_surrogate')[0]
#         data = float(subdir.split('_stealtype_')[0].split('data_')[1])
#         for file in files:
#             if 'log' in file:
#                 with open(os.path.join(subdir, file), 'r') as f:
#                     lines = [line.rstrip('\n') for line in f]
#                     best_accu = float(lines[-3].split('val_acc: ')[1].split(",")[0])
#                     best_fidelity = float(lines[-1].split('fidel_score: ')[1].split(",")[0])
#                     if layer == target_layer:
#                         print("{}, {}, {}, {:1.2f}/{:1.2f}".format(layer, exp_name, data, best_accu, best_fidelity))


# search_path = "./ideal_extraction_cifar100_GMrerun_largeLR"
# search_path = "./ideal_extraction_cifar10_correct"
# search_path = "./craft_ideal_100K"
# search_path = "./softtrain_prac_start120_5client"
# search_path = "./soft_ideal_100K"
search_path = "./vgg11_cifar100"
for target_layer in range(2, 9):
    for subdir, dirs, files in os.walk(search_path):
        if "stealtype" in subdir and "layer_-1" not in subdir:
            # print(subdir)
            layer = int(float(subdir.split('layer_')[1].split('_data')[0]))
            exp_name = subdir.split('stealtype_')[1].split('_epoch300_surrogate')[0]
            data = float(subdir.split('_stealtype_')[0].split('data_')[1])
            for file in files:
                if 'log' in file:
                    with open(os.path.join(subdir, file), 'r') as f:
                        lines = [line.rstrip('\n') for line in f]
                        # print(subdir)
                        best_accu = float(lines[-3].split('val_acc: ')[1].split(",")[0])
                        best_fidelity = float(lines[-1].split('fidel_score: ')[1].split(",")[0])
                        if layer == target_layer:
                            print("{}, {}, {}, {:1.2f}/{:1.2f}".format(layer, exp_name, data, best_accu, best_fidelity))