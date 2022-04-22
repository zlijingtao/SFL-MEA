#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128

num_client=2
num_epochs=200

random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

# regularization=gan_adv_step1
regularization=None
scheme=V2_epoch
ssim_threshold=0.5
regularization_strength_list="0.0"
folder_name="saves/baseline"
bottleneck_option=None
cutlayer_list="4"
num_client_list="1"
interval=1
train_gan_AE_type=custom
gan_loss_type=SSIM


# source_task_list="svhn mnist facescrub cifar10"
transfer_source_task=cifar10
dataset=cifar10
learning_rate=0.005 # 0.0005 for 7 & 8, 0.01 data proportion
local_lr_list="0.005"

attack_epochs=5
attack_client=0
num_query=10
attack_style="TrainME_option"
# data_proportion_list="0.2 0.01"
data_proportion_list="0.2"
train_clas_layer_list="5"
surrogate_arch="longer"
for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for cutlayer in $cutlayer_list; do
                        for num_client in $num_client_list; do
                                for local_lr in $local_lr_list; do
                                        for data_proportion in $data_proportion_list; do
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_V2_epoch_vgg11_bn_cutlayer_4_client_1_seed125_dataset_${dataset}_lr_0.05_200epoch
                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --test_best  --learning_rate=$learning_rate\
                                                        --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option} \
                                                        --attack_client=$attack_client  --num_query=$num_query  --regularization=$regularization  --regularization_strength=${regularization_strength} \
                                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer} --surrogate_arch=${surrogate_arch}
                                                done
                                        done
                                done
                        done
                done
        done
done