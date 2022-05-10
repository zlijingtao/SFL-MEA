#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=5
arch=vgg11_bn
batch_size=128
num_epochs=200
random_seed_list="125"
regularization=None
scheme=V2_epoch
regularization_strength_list="0.0"
folder_name="saves/baseline"
cutlayer="4"
num_client="1"

dataset=cifar10
learning_rate=0.005 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=200
attack_client=0
num_query_list="10000"
attack_style_list="SoftTrain_option"
# data_proportion_list="0.2 0.02"
data_proportion_list="0.02"
surrogate_arch_list="shorter longer thinner wider"
# train_clas_layer_list="2 3 4 5"
train_clas_layer_list="5"

for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for surrogate_arch in $surrogate_arch_list; do
                        for attack_style in $attack_style_list; do
                                for num_query in $num_query_list; do
                                        for data_proportion in $data_proportion_list; do
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_V2_epoch_${arch}_cutlayer_4_client_1_seed125_dataset_${dataset}_lr_0.05_200epoch

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --test_best  --learning_rate=$learning_rate\
                                                        --attack_epochs=$attack_epochs \
                                                        --attack_client=$attack_client  --num_query=$num_query  --regularization=$regularization  --regularization_strength=${regularization_strength} \
                                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer} --surrogate_arch=${surrogate_arch}
                                                done
                                        done
                                done
                        done
                done
        done
done