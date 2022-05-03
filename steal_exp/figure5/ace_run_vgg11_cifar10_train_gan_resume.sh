#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128

num_client=2
num_epochs=200

random_seed_list="125"

scheme=V2_epoch
ssim_threshold=0.5
regularization_strength_list="0.0"
folder_name="saves/train_attack"
transfer_source_task=cifar10
dataset=cifar10
learning_rate=0.005 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=300
attack_client=0
num_query=10
attack_style="Generator_option_nz256_resume"
regularization_list="gan_train_ME_nopoison_nz256_start160 gan_train_ME_nopoison_nz256_start120"
data_proportion_list="0.0"
train_clas_layer_list="2 3 4 5"
num_client_list="6 11"
cutlayer="4"
for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for regularization in $regularization_list; do
                        for num_client in $num_client_list; do
                                for data_proportion in $data_proportion_list; do
                                        for train_clas_layer in $train_clas_layer_list; do
                                        filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization}_${regularization_strength}_200epoch
                                        CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                --dataset=$dataset --scheme=$scheme  --learning_rate=$learning_rate\
                                                --attack_epochs=$attack_epochs \
                                                --attack_client=$attack_client  --num_query=$num_query  --regularization=$regularization  --regularization_strength=${regularization_strength} \
                                                --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer}
                                        done
                                done
                        done
                done
        done
done
