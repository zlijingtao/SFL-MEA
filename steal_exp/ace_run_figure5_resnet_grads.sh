#!/bin/bash
cd "$(dirname "$0")"

# bash figure5_resnet20/ace_run_resnet20_cifar10_train_craft.sh
# bash figure5_resnet20/ace_run_resnet20_cifar10_train_craft_resume.sh
# bash figure5_resnet20/ace_run_resnet20_cifar10_train_gan.sh
# bash figure5_resnet20/ace_run_resnet20_cifar10_train_gan_resume.sh
bash figure5_resnet20/ace_run_resnet20_cifar10_train_GM.sh
bash figure5_resnet20/ace_run_resnet20_cifar10_train_GM_resume.sh
bash figure5_resnet20/ace_run_resnet20_cifar10_train_soft.sh
bash figure5_resnet20/ace_run_resnet20_cifar10_train_soft_resume.sh