#!/bin/bash
cd "$(dirname "$0")"

# bash figure3_mob/ace_run_steal_attack_craftME_vgg11bn_cifar10.sh
# bash figure3_mob/ace_run_steal_attack_ganME_vgg11bn_cifar10.sh
bash figure3_mob/ace_run_steal_attack_GMME_vgg11bn_cifar10.sh
bash figure3_mob/ace_run_steal_attack_softME_vgg11bn_cifar10.sh
bash figure3_mob/ace_run_steal_attack_TrainME_vgg11bn_cifar10.sh