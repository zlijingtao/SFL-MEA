#!/bin/bash
cd "$(dirname "$0")"
# bash traditional_query_attack/ace_run_vgg11_baseline.sh
# bash traditional_query_attack/ace_run_steal_attack_jbda_gpu0.sh
bash traditional_query_attack/ace_run_steal_attack_knockoff_cifar100.sh
# bash traditional_query_attack/ace_run_steal_attack_knockoff_svhn.sh
# bash traditional_query_attack/ace_run_steal_attack_gpu0_noiid.sh
# bash traditional_query_attack/ace_run_steal_attack_cifar100_gpu0.sh