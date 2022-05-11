#!/bin/bash
cd "$(dirname "$0")"

bash table2/ace_run_steal_attack_craftME_vgg11bn_cifar10.sh
bash table2/ace_run_steal_attack_ganME_vgg11bn_cifar10.sh
bash table2/ace_run_steal_attack_GMME_vgg11bn_cifar10.sh
bash table2/ace_run_vgg11_cifar10_train_craft.sh
bash table2/ace_run_vgg11_cifar10_train_craft_resume.sh
bash table2/ace_run_vgg11_cifar10_train_gan.sh
bash table2/ace_run_vgg11_cifar10_train_gan_resume.sh
bash table2/ace_run_vgg11_cifar10_train_GM.sh
bash table2/ace_run_vgg11_cifar10_train_GM_resume.sh