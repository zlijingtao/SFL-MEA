#!/bin/bash
cd "$(dirname "$0")"

bash offline/run_craftME_vgg11bn_cifar10.sh
bash offline/run_ganME_vgg11bn_cifar10.sh
bash offline/run_GMME_vgg11bn_cifar10.sh
bash offline/run_softrainME_vgg11bn_cifar10.sh
bash offline/run_TrainME_vgg11bn_cifar10.sh
