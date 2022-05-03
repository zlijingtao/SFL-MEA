#!/bin/bash
cd "$(dirname "$0")"
cd ../../
GPU_id=0
arch=vgg11_bn
batch_size=128

num_epochs=200
dataset=cifar10
scheme=V2_epoch
random_seed_list="125"
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}
learning_rate=0.005

num_client=1
folder_name="saves/regu"
attack_epochs=300
attack_client=0
num_query_list="1"
attack_style_list="TrainME_option"
data_proportion_list="0.02 0.2"
regularization_list=(nopeek_inv nopeek_inv nopeek_inv nopeek_inv)
regularization_strength_list=(0.5 1.0 2.0 5.0)
cutlayer_list="9 11"
train_clas_layer_list="5"
for random_seed in $random_seed_list; do
        for cutlayer in $cutlayer_list; do
                for attack_style in $attack_style_list; do
                        for num_query in $num_query_list; do
                                for data_proportion in $data_proportion_list; do
                                        for index in ${!regularization_list[*]}; do 
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization_list[$index]}_${regularization_strength_list[$index]}_${num_epochs}epoch

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --test_best  --learning_rate=$learning_rate\
                                                        --attack_epochs=$attack_epochs \
                                                        --attack_client=$attack_client  --num_query=$num_query \
                                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer}
                                                done
                                        done
                                done
                        done
                done
        done
done

cutlayer_list="11"
train_clas_layer_list="4"
for random_seed in $random_seed_list; do
        for cutlayer in $cutlayer_list; do
                for attack_style in $attack_style_list; do
                        for num_query in $num_query_list; do
                                for data_proportion in $data_proportion_list; do
                                        for index in ${!regularization_list[*]}; do 
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization_list[$index]}_${regularization_strength_list[$index]}_${num_epochs}epoch

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_model_steal.py   --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --folder ${folder_name} --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --test_best  --learning_rate=$learning_rate\
                                                        --attack_epochs=$attack_epochs \
                                                        --attack_client=$attack_client  --num_query=$num_query \
                                                        --attack_style=$attack_style  --data_proportion=${data_proportion} --train_clas_layer=${train_clas_layer}
                                                done
                                        done
                                done
                        done
                done
        done
done