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
data_proportion_list="0.02"
regularization=l1
regularization_strength_list="5e-4 2e-4 1e-4 5e-5"
cutlayer_list="9"
train_clas_layer_list="5"
for random_seed in $random_seed_list; do
        for cutlayer in $cutlayer_list; do
                for attack_style in $attack_style_list; do
                        for num_query in $num_query_list; do
                                for data_proportion in $data_proportion_list; do
                                        for regularization_strength in $regularization_strength_list; do 
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization}_${regularization_strength}_${num_epochs}epoch

                                                target_client=0
                                                attack_scheme=MIA
                                                attack_epochs=50
                                                average_time=1

                                                internal_C=64
                                                N=4
                                                test_gan_AE_type=res_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_client=${target_client} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                
                                                
                                                
                                                
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
                                        for regularization_strength in $regularization_strength_list; do 
                                                for train_clas_layer in $train_clas_layer_list; do
                                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_0.05_${regularization}_${regularization_strength}_${num_epochs}epoch

                                                target_client=0
                                                attack_scheme=MIA
                                                attack_epochs=50
                                                average_time=1

                                                internal_C=64
                                                N=4
                                                test_gan_AE_type=res_normN${N}C${internal_C}

                                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                                        --dataset=$dataset --scheme=$scheme --target_client=${target_client} --test_best\
                                                        --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs\
                                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                                
                                                
                                                done
                                        done
                                done
                        done
                done
        done
done