
cd "$(dirname "$0")"
GPU_id=3
lr=0.01
data_proportion_list="0.2"
train_clas_layer_list="3 4"
# train_clas_layer_list="3 4 5 6 7 8"
for data_proportion in $data_proportion_list; do
    for train_clas_layer in $train_clas_layer_list; do
        CUDA_VISIBLE_DEVICES=${GPU_id} python imagenet.py \
            -a mobilenetv2 \
            -d ../../../imagenet \
            --weight pretrained/mobilenetv2_1.0-0c6065bc.pth \
            --width-mult 1.0 \
            --input-size 224  --lr=${lr}\
            -e --epochs 100 --train_clas_layer=${train_clas_layer} --data-backend pytorch --data_proportion=${data_proportion}
    done
done