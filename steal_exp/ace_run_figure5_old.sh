#!/bin/bash
cd "$(dirname "$0")"

# bash figure5_old/ace_run_vgg11_cifar10_train_GM.sh
# bash figure5_old/ace_run_vgg11_cifar10_train_GM_resume.sh
bash figure5_old/ace_run_vgg11_cifar10_train_soft.sh
bash figure5_old/ace_run_vgg11_cifar10_train_softtrain_resume.sh