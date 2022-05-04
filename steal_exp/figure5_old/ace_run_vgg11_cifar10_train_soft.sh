#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=5
arch=vgg11_bn
batch_size=128
num_epochs=200
dataset_list="cifar10"
scheme=V2_epoch
random_seed_list="125"
regularization_list="soft_train_ME_start120"
learning_rate=0.05
regularization_strength=0.0
cutlayer_list="4"
# num_client_list="5 10"
num_client_list="10"
folder_name="saves/train_attack_old"
# ace_V2_epoch_vgg11_bn_cutlayer_4_client_1_seed125_dataset_cifar10_lr_0.05_None_both_custom_0.0_200epoch_bottleneck_None
for dataset in $dataset_list; do
        for random_seed in $random_seed_list; do
                for cutlayer in $cutlayer_list; do
                        for regularization in $regularization_list; do
                                for num_client in $num_client_list; do
                                        filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_${regularization_strength}_${num_epochs}epoch
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate\
                                                --folder ${folder_name}
                                done
                        done
                done
        done
done
regularization_list="soft_train_ME_start160"
num_client_list="5 10"
folder_name="saves/train_attack_old"
# ace_V2_epoch_vgg11_bn_cutlayer_4_client_1_seed125_dataset_cifar10_lr_0.05_None_both_custom_0.0_200epoch_bottleneck_None
for dataset in $dataset_list; do
        for random_seed in $random_seed_list; do
                for cutlayer in $cutlayer_list; do
                        for regularization in $regularization_list; do
                                for num_client in $num_client_list; do
                                        filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_${regularization_strength}_${num_epochs}epoch
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate\
                                                --folder ${folder_name}
                                done
                        done
                done
        done
done