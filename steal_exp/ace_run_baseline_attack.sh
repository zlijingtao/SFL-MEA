#!/bin/bash
cd "$(dirname "$0")"
# bash baseline_attack/ace_run_vgg11_baseline.sh
bash baseline_attack/ace_run_steal_attack_gpu0.sh
# bash baseline_attack/ace_run_steal_attack_gpu0_noiid.sh
# bash baseline_attack/ace_run_steal_attack_cifar100_gpu0.sh