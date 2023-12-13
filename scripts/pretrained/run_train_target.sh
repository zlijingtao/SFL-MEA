cd "$(dirname "$0")"
cd ../../

python main.py --filename vgg11-cifar10 --dataset cifar10 --learning_rate 0.05 --scheme V1

# python main.py --filename resnet20-cifar10-v1 --dataset cifar10 --arch resnet20 --learning_rate 0.02 --scheme V1

# python main.py --filename resnet18-cifar10-v1 --dataset cifar10 --arch resnet18 --learning_rate 0.02 --scheme V1

# python main.py --filename mobilenetv2-cifar10-v1-cut4 --dataset cifar10 --arch mobilenetv2 --learning_rate 0.02 --scheme V1

# python main.py --filename vgg13-cifar10-v1 --dataset cifar10 --arch vgg13_bn --learning_rate 0.02 --scheme V1

# python main.py --filename vgg11-cifar100 --dataset cifar100 --learning_rate 0.05 --scheme V1

# python main.py --filename vgg11-svhn --dataset svhn --learning_rate 0.05 --scheme V1

# python main.py --filename vgg11-imagenet12-new-v1 --dataset imagenet12 --learning_rate 0.05 --scheme V1
