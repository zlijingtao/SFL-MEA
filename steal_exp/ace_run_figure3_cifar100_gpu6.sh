#!/bin/bash
cd "$(dirname "$0")"

bash figure3/ace_run_steal_attack_GMME_vgg11bn_cifar100.sh
bash figure3/ace_run_steal_attack_softME_vgg11bn_cifar100.sh