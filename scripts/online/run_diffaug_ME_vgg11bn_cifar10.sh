#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128
num_epochs=200
random_seed_list="123"

scheme=V2
dataset=cifar10
learning_rate=0.02 # 0.00005 for 7 & 8, 0.01 data proportion
learning_rate_MEA=0.001
attack_epochs=50
attack_client=0
# regularization_list="gan_assist_train_ME_start0 naive_train_ME_diffaug_start0"
# regularization_list="gan_assist_train_ME_reverse_grad_start0 naive_train_ME_diffaug_start0"
regularization_list="naive_train_ME_half_half_diffaug_start0"
cutlayer_list="10"
regularization_strength_list="1.0"
num_client_list="5"
noniid_ratio_list="1.0"
# last_client_fix_amount=1000
last_client_fix_amount_list="50 200"
for random_seed in $random_seed_list; do
        for num_client in $num_client_list; do
                for cutlayer in $cutlayer_list; do
                        for noniid_ratio in $noniid_ratio_list; do
                                for regularization in $regularization_list; do
                                        for last_client_fix_amount in $last_client_fix_amount_list; do
                                                for regularization_strength in $regularization_strength_list; do
                                        
                                                folder_name="saves/adv"
                                                filename="vgg11-cifar10-$regularization-str$regularization_strength-cut$cutlayer-client$num_client-noniid$noniid_ratio--data$last_client_fix_amount"
                                                CUDA_VISIBLE_DEVICES=$GPU_id python main_steal_online.py   --arch=$arch --cutlayer=$cutlayer --batch_size=$batch_size \
                                                        --folder $folder_name --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --noniid_ratio=$noniid_ratio --scheme=$scheme  --learning_rate=$learning_rate --learning_rate_MEA=$learning_rate_MEA --attack_epochs=$attack_epochs \
                                                        --attack_client=$attack_client  --regularization=$regularization  --regularization_strength=$regularization_strength --last_client_fix_amount=$last_client_fix_amount
                                                
                                                done
                                        done
                                done
                        done
                done
        done
done