#!/bin/bash
cd "$(dirname "$0")"



bash analysis/run_ganassistME_vgg11bn_cifar10_baseline_running_margin.sh
bash analysis/run_ganassistME_vgg11bn_cifar10_proposed_running_margin.sh
# bash analysis/run_ganME_vgg11bn_cifar10_baseline_running_margin.sh
# bash analysis/run_ganME_vgg11bn_cifar10_proposed_running_margin.sh
# bash analysis/run_naive_ME_vgg11bn_cifar10_running_margin.sh
# bash analysis/run_naive_ME_vgg11bn_cifar10_grad_stats.sh