#!/bin/bash
cd "$(dirname "$0")"

bash figure4/ace_run_steal_attack_GMME_vgg11bn_cifar10_thinner.sh
bash figure4/ace_run_steal_attack_GMME_vgg11bn_cifar10_wider.sh
bash figure4/ace_run_steal_attack_TrainME_vgg11bn_cifar10_thinner.sh
bash figure4/ace_run_steal_attack_TrainME_vgg11bn_cifar10_wider.sh