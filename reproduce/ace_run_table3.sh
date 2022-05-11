#!/bin/bash
cd "$(dirname "$0")"

bash table2/ace_run_steal_attack_softME_vgg11bn_cifar10.sh
bash table2/ace_run_steal_attack_TrainME_vgg11bn_cifar10.sh
bash table2/ace_run_vgg11_cifar10_train_softtrain.sh
bash table2/ace_run_vgg11_cifar10_train_softtrain_resume.sh