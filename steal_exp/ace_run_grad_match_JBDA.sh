#!/bin/bash
cd "$(dirname "$0")"
bash gradient_matching/ace_run_matching_attack_cifar10_gpu0_JBDA.sh
bash gradient_matching/ace_run_matching_attack_cifar100_gpu0_JBDA.sh