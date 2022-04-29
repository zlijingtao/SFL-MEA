#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128

num_client=2
num_epochs=200
dataset_list="cifar10"
scheme=V2_epoch
random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}
learning_rate=0.05
cutlayer_list="9 11"
num_client_list="1"
folder_name="saves/regu"
regularization_list=(gradient_noise_cloud topkprune dropout local_dp_laplace nopeek l1 l2 gradient_noise_local)
regularization_strength_list=(1.0 0.75 0.5 20.0 1.0 1e-3 1e-4 1.0)
for dataset in $dataset_list; do
        for random_seed in $random_seed_list; do
                for cutlayer in $cutlayer_list; do
                        for num_client in $num_client_list; do
                                for index in ${!regularization_list[*]}; do 
                                        filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization_list[$index]}_${regularization_strength_list[$index]}_${num_epochs}epoch
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme --regularization=${regularization_list[$index]} --regularization_strength=${regularization_strength_list[$index]}\
                                                --random_seed=$random_seed --learning_rate=$learning_rate \
                                                --folder ${folder_name}
                                done
                        done
                done
        done
done
