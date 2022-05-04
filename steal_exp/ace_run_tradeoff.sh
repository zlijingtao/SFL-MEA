#!/bin/bash
cd "$(dirname "$0")"

bash tradeoff/ace_run_vgg11_baseline_true_cutlayer.sh
bash tradeoff/ace_run_steal_attack_TrainME_vgg11bn_cifar10_true_cutlayer.sh