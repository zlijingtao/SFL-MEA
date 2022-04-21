#!/bin/bash
cd "$(dirname "$0")"

bash figure3/ace_run_steal_attack_craftME_vgg11bn_cifar100.sh
bash figure3/ace_run_steal_attack_ganME_vgg11bn_cifar100.sh