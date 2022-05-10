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
random_seed_list="123"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}
learning_rate=0.02
cutlayer_list="9 11"
num_client_list="1"
folder_name="saves/regu1"
regularization_list=(l1 l1 l1 l1 l1)
regularization_strength_list=(0.0 5e-4 2e-4 1e-4 5e-5)
# regularization_list=(l2 l2 l2)
# regularization_strength_list=(5e-1 2e-1 1e-1)
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
