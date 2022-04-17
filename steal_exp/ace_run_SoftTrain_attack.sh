#!/bin/bash
cd "$(dirname "$0")"
bash baseline_attack/ace_run_steal_attack_gpu0.sh

bash SoftTrain/ace_run_SoftTrain_attack_cifar10_gpu0.sh

# bash SoftTrain/ace_run_SoftTrain_attack_cifar100_gpu0.sh