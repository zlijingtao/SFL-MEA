#!/bin/bash
cd "$(dirname "$0")"

# run attack without data vgg11-cifar10-iid
bash online/run_craftME_vgg11bn_cifar10_surrogate.sh
bash online/run_ganME_vgg11bn_cifar10_baseline.sh
bash online/run_ganME_vgg11bn_cifar10_proposed.sh

# run attack with data vgg11-cifar10-iid
bash online/run_naive_ME_vgg11bn_cifar10.sh
bash online/run_ganassistME_vgg11bn_cifar10_baseline.sh
bash online/run_ganassistME_vgg11bn_cifar10_proposed.sh

# run attack with data vgg11-cifar10-noniid
bash online/run_naive_ME_vgg11bn_cifar10_noniid.sh
bash online/run_ganassistME_vgg11bn_cifar10_proposed_noniid.sh

# run proposed attack with data vgg11-cifar100
bash online/run_ganassistME_vgg11bn_cifar100_proposed.sh

