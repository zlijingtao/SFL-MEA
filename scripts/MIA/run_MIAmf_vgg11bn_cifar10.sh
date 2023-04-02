#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128
num_epochs=200
regularization=None
scheme=V2
folder_name="saves/baseline"
filename="vgg11-cifar10"
num_client="1"
dataset=cifar10
aux_dataset_name=cifar10
attack_scheme=MIA_mf
attack_loss_type=MSE
MIA_optimizer=Adam
MIA_lr=0.5
gan_loss_type=SSIM
attack_epochs=50
select_layer_output="-1"
# random_seed_list="123 124"
random_seed_list="123"
# cutlayer_list="4 5 6 7 8"
cutlayer_list="6 7 8 9 10"

for random_seed in $random_seed_list; do
        for cutlayer in $cutlayer_list; do
                CUDA_VISIBLE_DEVICES=$GPU_id python main_model_inversion.py   --arch=$arch --cutlayer=$cutlayer --batch_size=$batch_size \
                        --folder $folder_name --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                        --dataset=$dataset --scheme=$scheme  --random_seed=$random_seed --attack_scheme=$attack_scheme --aux_dataset_name=$aux_dataset_name\
                        --MIA_optimizer=$MIA_optimizer --MIA_lr=$MIA_lr --attack_epochs=$attack_epochs  --attack_loss_type=$attack_loss_type --gan_loss_type=$gan_loss_type\
                        --select_layer_output=$select_layer_output
        done
done
