GPU_id=0
arch=vgg11_bn
batch_size=128

num_client=2
num_epochs=200
dataset_list="mnist"
scheme=V2_epoch
random_seed=125
#Extra argement (store_true): --collude_use_public, --initialize_different  --collude_not_regularize  --collude_not_regularize --num_client_regularize ${num_client_regularize}

regularization=None
learning_rate=0.05
local_lr=-1
regularization_strength_list="0.0"
cutlayer_list="4"
num_client_list="2 5"
train_gan_AE_type=custom
gan_loss_type=SSIM
folder_name="poison_saves/mnist/vanilla_poison_shuffle_reject_aggregation"
bottleneck_option=None

for dataset in $dataset_list; do
        for regularization_strength in $regularization_strength_list; do
                for cutlayer in $cutlayer_list; do
                        for num_client in $num_client_list; do
                                filename=ace_${scheme}_${arch}_cutlayer_${cutlayer}_client_${num_client}_seed${random_seed}_dataset_${dataset}_lr_${learning_rate}_${regularization}_both_${train_gan_AE_type}_${regularization_strength}_${num_epochs}epoch_bottleneck_${bottleneck_option}
                                
                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_MIA.py --arch=${arch}  --cutlayer=$cutlayer --batch_size=${batch_size} \
                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                        --dataset=$dataset --scheme=$scheme --regularization=${regularization} --regularization_strength=${regularization_strength}\
                                        --random_seed=$random_seed --learning_rate=$learning_rate --gan_AE_type ${train_gan_AE_type} --gan_loss_type ${gan_loss_type}\
                                        --local_lr $local_lr --bottleneck_option ${bottleneck_option} --folder ${folder_name}

                                target_client=0
                                attack_scheme=MIA
                                attack_epochs=50
                                average_time=1

                                internal_C=64
                                N=4
                                test_gan_AE_type=res_normN${N}C${internal_C}

                                CUDA_VISIBLE_DEVICES=${GPU_id} python main_test_MIA.py --arch=${arch} --cutlayer=$cutlayer --batch_size=${batch_size} \
                                        --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                                        --dataset=$dataset --scheme=$scheme --target_client=${target_client} \
                                        --attack_scheme=$attack_scheme --attack_scheme=$attack_scheme --attack_epochs=$attack_epochs  --bottleneck_option ${bottleneck_option}\
                                        --average_time=$average_time --gan_AE_type ${test_gan_AE_type} --regularization=$regularization  --regularization_strength=${regularization_strength} --folder ${folder_name}
                                
                        done
                done
        done
done