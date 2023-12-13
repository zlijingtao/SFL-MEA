#!/bin/bash
cd "$(dirname "$0")"

# get pretrained vgg11-cifar10 model
bash pretrained/run_train_target.sh

# run offline ME attack
bash offline/run_craftME_vgg11bn_cifar10.sh
bash offline/run_ganME_vgg11bn_cifar10.sh
bash offline/run_GMME_vgg11bn_cifar10.sh
bash offline/run_softrainME_vgg11bn_cifar10.sh
bash offline/run_TrainME_vgg11bn_cifar10.sh
