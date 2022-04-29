#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=mobilenetv2
batch_size=128

num_client=2
num_epochs=200

random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

# regularization=gan_adv_step1

scheme=V2_epoch
ssim_threshold=0.5
regularization_strength="1.0"
folder_name="saves/train_attack"

# source_task_list="svhn mnist facescrub cifar10"
transfer_source_task=cifar10
dataset=cifar10
learning_rate=0.005 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=100
attack_client=0
num_query=10
attack_style="GM_option_resume"
regularization_list="GM_train_ME_CIFAR100_start120 GM_train_ME_CIFAR100_start160"
data_proportion_list="0.1"
num_client_list="6 11"
cutlayer_list=(18 17 16 15)
train_clas_layer_list=(2 3 4 5)
for random_seed in $random_seed_list; do
        for regularization in $regularization_list; do
                for num_client in $num_client_list; do
                        for data_proportion in $data_proportion_list; do
                                for index in ${!cutlayer_list[*]}; do 
                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer_list[$index]}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.02_${regularization}_${regularization_strength}_200epoch
                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=${cutlayer_list[$index]} --batch_size=${batch_size} \
                                        --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                        --dataset=$dataset --scheme=$scheme  --learning_rate=$learning_rate\
                                        --attack_epochs=$attack_epochs \
                                        --attack_client=$attack_client  --num_query=$num_query  --regularization=$regularization  --regularization_strength=${regularization_strength} \
                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer_list[$index]}
                                done
                        done
                done
        done
done
