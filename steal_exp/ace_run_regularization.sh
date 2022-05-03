#!/bin/bash
cd "$(dirname "$0")"
# bash regularization/ace_run_vgg11_baseline_regularization.sh
# bash regularization/ace_run_steal_attack_TrainME_vgg11bn_cifar10_l1_regu.sh

bash regularization/ace_run_vgg11_baseline_nopeek_inv.sh
bash regularization/ace_run_steal_attack_TrainME_vgg11bn_cifar10_inv_nopeek_regu.sh