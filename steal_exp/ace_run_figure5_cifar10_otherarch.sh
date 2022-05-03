#!/bin/bash
cd "$(dirname "$0")"

bash figure5_cifar10_otherarch/ace_run_vgg11_cifar10_train_GM.sh
bash figure5_cifar10_otherarch/ace_run_vgg11_cifar10_train_GM_resume.sh
bash figure5_cifar10_otherarch/ace_run_vgg11_cifar10_train_soft.sh
bash figure5_cifar10_otherarch/ace_run_vgg11_cifar10_train_soft_resume.sh