#!/bin/bash
cd "$(dirname "$0")"

bash figure4/ace_run_steal_attack_craftME_vgg11bn_cifar10_thinner.sh
bash figure4/ace_run_steal_attack_craftME_vgg11bn_cifar10_wider.sh
bash figure4/ace_run_steal_attack_ganME_vgg11bn_cifar10_thinner.sh
bash figure4/ace_run_steal_attack_ganME_vgg11bn_cifar10_wider.sh
bash figure4/ace_run_steal_attack_craftME_vgg11bn_cifar10_longer.sh
bash figure4/ace_run_steal_attack_craftME_vgg11bn_cifar10_shorter.sh
bash figure4/ace_run_steal_attack_ganME_vgg11bn_cifar10_longer.sh
bash figure4/ace_run_steal_attack_ganME_vgg11bn_cifar10_shorter.sh