#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch_list="vgg11_bn"
batch_size=128

num_client=2
num_epochs=200

random_seed="125"

scheme=V2_epoch
ssim_threshold=0.5
regularization_strength="0.0"
folder_name="saves/train_attack"

transfer_source_task=cifar10
dataset_list="cifar10"
learning_rate=0.02 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=300
attack_client=0
num_query=10
attack_style_list="TrainME_option"
regularization_list="None"
data_proportion_list="0.2"
num_client_list="5 10"
cutlayer_list=(4 4 4 4 4 4 4)
train_clas_layer_list=(2 3 4 5 6 7 8)
for attack_style in $attack_style_list; do
        for arch in $arch_list; do
                for dataset in $dataset_list; do
                        for regularization in $regularization_list; do
                                for num_client in $num_client_list; do
                                        for data_proportion in $data_proportion_list; do
                                                for index in ${!cutlayer_list[*]}; do 
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer_list[$index]}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization}_${regularization_strength}_200epoch
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
        done
done