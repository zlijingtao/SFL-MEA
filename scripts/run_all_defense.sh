#!/bin/bash
cd "$(dirname "$0")"

# bash defense/run_l1.sh
# bash defense/run_ganME_gradient_noise.sh

bash pretrained/run_train_target.sh
bash pretrained/run_train_other_datasets.sh

bash defense/run_l2.sh
# bash offline/run_GMME_vgg11bn_cifar10.sh
# bash offline/run_softrainME_vgg11bn_cifar10.sh
# bash offline/run_TrainME_vgg11bn_cifar10.sh
