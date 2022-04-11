#!/bin/bash
cd "$(dirname "$0")"
# bash baseline_attack/ace_run_vgg11_baseline.sh
bash gan_attack/ace_run_gan_attack_cifar10.sh
# bash gan_attack/ace_run_gan_attack_cifar100.sh
# bash baseline_attack/ace_run_steal_attack_cifar100_gpu0.sh