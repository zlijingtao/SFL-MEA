#!/bin/bash
cd "$(dirname "$0")"

bash noniid_TrainME/ace_run_steal_attack_TrainME_vgg11bn_cifar10.sh
bash noniid_TrainME/ace_run_steal_attack_TrainME_vgg11bn_cifar100.sh