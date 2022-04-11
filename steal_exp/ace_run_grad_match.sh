#!/bin/bash
cd "$(dirname "$0")"
bash gradient_matching/ace_run_steal_attack_gradmatch_cifar100.sh
bash gradient_matching/ace_run_steal_attack_gradmatch_svhn.sh