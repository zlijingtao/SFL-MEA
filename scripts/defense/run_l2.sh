cd "$(dirname "$0")"
cd ../../

arch=vgg11_bn
batch_size=128
num_epochs=200
regularization=None
scheme=V2
regularization_strength_list="0.0"
folder_name="saves/defense"
num_client="1"

dataset=cifar10
learning_rate=0.02 # 0.00005 for 7 & 8, 0.01 data proportion

attack_epochs=200
attack_client=0

num_query_list="100000"
attack_style_list="SoftTrain_option TrainME_option"
data_proportion_list="0.02"

regularization_strength_list="0.2 0.15 0.05"

for regularization_strength in $regularization_strength_list; do

    filename="l2-$regularization_strength-vgg11-cifar10-cut-5"

    python main.py --regularization l2 --regularization_strength=$regularization_strength  --folder=$folder_name --filename=$filename --cutlayer 10
    
    for num_query in $num_query_list; do
        for attack_style in $attack_style_list; do
            for data_proportion in $data_proportion_list; do
            python main_steal_offline.py   --arch=vgg11_bn --cutlayer=10 --batch_size=$batch_size \
                    --folder $folder_name --filename=$filename --num_client=$num_client --num_epochs=$num_epochs \
                    --dataset=$dataset --scheme=$scheme  --learning_rate=$learning_rate --attack_epochs=$attack_epochs \
                    --attack_client=$attack_client  --num_query=$num_query  --regularization=None \
                    --attack_style=$attack_style  --data_proportion=$data_proportion --train_clas_layer=5

            done
        done
    done
done