#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=mobilenetv2
batch_size=128
num_epochs=200
random_seed_list="125"
regularization=None
scheme=V2_epoch
regularization_strength_list="0.0"
folder_name="saves/baseline"
cutlayer_list="4"
num_client="1"

dataset=cifar10
learning_rate=0.02 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=300
attack_client=0
num_query_list="100000"
attack_style_list="Generator_option"
data_proportion_list="0.0"

# train_clas_layer_list="2 3 4 5"
train_clas_layer_list="2 3 4 5"

for random_seed in $random_seed_list; do
        for regularization_strength in $regularization_strength_list; do
                for cutlayer in $cutlayer_list; do
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
                                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer}
                                                done
                                        done
                                done
                        done
                done
        done
done

# learning_rate=0.05

# train_clas_layer_list="6 7 8"

# for random_seed in $random_seed_list; do
#         for regularization_strength in $regularization_strength_list; do
#                 for cutlayer in $cutlayer_list; do
#                         for attack_style in $attack_style_list; do
#                                 for num_query in $num_query_list; do
#                                         for data_proportion in $data_proportion_list; do
#                                                 for train_clas_layer in $train_clas_layer_list; do
#                                                 filename=ace_V2_epoch_vgg11_bn_cutlayer_4_client_1_seed125_dataset_${dataset}_lr_0.05_200epoch
                                                
#                                                 CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
#                                                         --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
#                                                         --dataset=$dataset --scheme=$scheme --test_best  --learning_rate=$learning_rate\
#                                                         --attack_epochs=$attack_epochs \
#                                                         --attack_client=$attack_client  --num_query=$num_query  --regularization=$regularization  --regularization_strength=${regularization_strength} \
#                                                         --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer}
#                                                 done
#                                         done
#                                 done
#                         done
#                 done
#         done
# done