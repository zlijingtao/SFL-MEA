#!/bin/bash
cd "$(dirname "$0")"

bash attack/run_withdata_EMGAN_vgg11bn_cifar10.sh
bash attack/run_withdata_TrainME_vgg11bn_cifar10.sh
bash attack/run_datafree_EMGAN_vgg11bn_cifar10.sh
bash attack/run_datafree_ganME_vgg11bn_cifar10.sh


