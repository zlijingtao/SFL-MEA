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
attack_epochs=200
attack_client=0

regularization=soft_train_ME_start150
cutlayer_list="10 11 12 13"
# cutlayer_list="14"
regularization_strength_list="1.0"
num_client_list="5"
noniid_ratio_list="1.0"
last_n_batch_list="10 50 100 500"
for random_seed in $random_seed_list; do
        for num_client in $num_client_list; do
                for cutlayer in $cutlayer_list; do
                        for noniid_ratio in $noniid_ratio_list; do
                                for regularization_strength in $regularization_strength_list; do
                                        for last_n_batch in $last_n_batch_list; do
                                        folder_name="saves/train-ME"
                                        filename="vgg11-cifar10-$regularization-step$regularization_strength-cut$cutlayer-client$num_client-noniid$noniid_ratio"
                                        CUDA_VISIBLE_DEVICES=$GPU_id python main_steal_online.py   --arch=$arch --cutlayer=$cutlayer --batch_size=$batch_size \
                                                --folder $folder_name --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --noniid_ratio=$noniid_ratio --scheme=$scheme  --learning_rate=$learning_rate --attack_epochs=$attack_epochs \
                                                --attack_client=$attack_client  --regularization=$regularization  --regularization_strength=$regularization_strength --last_n_batch=$last_n_batch
                                        done
                                done
                        done
                done
        done
done
