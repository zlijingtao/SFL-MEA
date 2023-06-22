#!/bin/bash
cd "$(dirname "$0")"



bash mixup_comparison/run_ganassistME_vgg11bn_cifar10_mixup.sh
bash mixup_comparison/run_ganassistME_vgg11bn_cifar10_mix_sameclass.sh
bash mixup_comparison/run_ganassistME_vgg11bn_cifar10_mix_randomnoise.sh
# bash analysis/run_ganME_vgg11bn_cifar10_baseline_running_margin.sh
# bash analysis/run_ganME_vgg11bn_cifar10_proposed_running_margin.sh
# bash analysis/run_naive_ME_vgg11bn_cifar10_running_margin.sh
# bash analysis/run_naive_ME_vgg11bn_cifar10_grad_stats.sh