#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128
num_epochs=200
random_seed_list="125"

scheme=V2
dataset=cifar10
learning_rate=0.02 # 0.00005 for 7 & 8, 0.01 data proportion
learning_rate_MEA=0.001
attack_epochs=200
attack_client=0

regularization=naive_train_ME_start0
# cutlayer_list="10 11 12 13"
# cutlayer_list="11 12 13"
cutlayer_list="10"
regularization_strength=1.0
# num_client_list="5 10"
num_client_list="5"
noniid_ratio_list="1.0"
last_n_batch_list="0"
# last_client_fix_amount_list="200 500 1000 2000"
last_client_fix_amount_list="50 100 200"
for random_seed in $random_seed_list; do
        for num_client in $num_client_list; do
                for cutlayer in $cutlayer_list; do
                        for noniid_ratio in $noniid_ratio_list; do
                                for last_client_fix_amount in $last_client_fix_amount_list; do
                                        for last_n_batch in $last_n_batch_list; do
                                        folder_name="saves/train-ME-seed125"
                                        filename="vgg11-cifar10-$regularization-step$regularization_strength-cut$cutlayer-client$num_client-noniid$noniid_ratio--data$last_client_fix_amount"
                                        CUDA_VISIBLE_DEVICES=$GPU_id python main_steal_online.py   --arch=$arch --cutlayer=$cutlayer --batch_size=$batch_size \
                                                --folder $folder_name --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --noniid_ratio=$noniid_ratio --scheme=$scheme  --learning_rate=$learning_rate --learning_rate_MEA=$learning_rate_MEA --attack_epochs=$attack_epochs \
                                                --attack_client=$attack_client  --regularization=$regularization  --regularization_strength=$regularization_strength --last_n_batch=$last_n_batch --last_client_fix_amount=$last_client_fix_amount
                                                
                                        done
                                done
                        done
                done
        done
done
