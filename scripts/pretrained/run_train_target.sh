cd "$(dirname "$0")"
cd ../../

python main.py --filename vgg11-cifar10-new

python main.py --filename vgg11-cifar10-new-v1 --dataset cifar10 --learning_rate 0.05 --scheme V1