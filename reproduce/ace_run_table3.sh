#!/bin/bash
cd "$(dirname "$0")"

# bash table3/ace_run_vgg11_cifar10_train_craft.sh
# bash table3/ace_run_vgg11_cifar10_train_craft_resume.sh
# bash table3/ace_run_vgg11_cifar10_train_gan.sh
# bash table3/ace_run_vgg11_cifar10_train_gan_resume.sh
bash table3/ace_run_vgg11_cifar10_train_train.sh
bash table3/ace_run_vgg11_cifar10_train_train_resume.sh
# bash table3/ace_run_vgg11_cifar10_train_GM.sh
bash table3/ace_run_vgg11_cifar10_train_GM_resume.sh
# bash table3/ace_run_vgg11_cifar10_train_softtrain.sh
# bash table3/ace_run_vgg11_cifar10_train_softtrain_resume.sh
