#!/bin/bash
cd "$(dirname "$0")"

# bash train_gan_attack/ace_run_vgg11_cifar10_train_gan.sh
bash train_attack/ace_run_vgg11_cifar10_train_GM.sh
bash train_attack/ace_run_vgg11_cifar10_train_normal.sh
bash train_attack/ace_run_vgg11_cifar10_train_soft.sh