cd "$(dirname "$0")"
cd ../../

# MNIST/FMNIST/SVHN


python main.py --filename vgg11-cifar100 --dataset cifar100 --learning_rate 0.05
python main.py --filename vgg11-mnist --dataset mnist --learning_rate 0.02
python main.py --filename vgg11-svhn --dataset svhn --learning_rate 0.02
python main.py --filename vgg11-fmnist --dataset fmnist --learning_rate 0.02
python main.py --filename vgg11-femnist --dataset femnist --learning_rate 0.02

python main.py --filename vgg11-cifar100-v1 --dataset cifar100 --learning_rate 0.05 --scheme V1
python main.py --filename vgg11-mnist-v1 --dataset mnist --learning_rate 0.02 --scheme V1
python main.py --filename vgg11-svhn-v1 --dataset svhn --learning_rate 0.02 --scheme V1
python main.py --filename vgg11-fmnist-v1 --dataset fmnist --learning_rate 0.02 --scheme V1
python main.py --filename vgg11-femnist-v1 --dataset femnist --learning_rate 0.02 --scheme V1

