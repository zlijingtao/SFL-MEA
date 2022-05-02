#!/bin/bash
cd "$(dirname "$0")"
# bash figure6/ace_run_vgg11_baseline_otherdata.sh
# bash figure6/ace_run_steal_attack_TrainME_vgg11bn_mnist.sh
# bash figure6/ace_run_steal_attack_TrainME_vgg11bn_fmnist.sh
# bash figure6/ace_run_steal_attack_TrainME_vgg11bn_svhn.sh



# bash figure6/ace_run_resnet20_baseline_otherdata.sh
# bash figure6/ace_run_mobilenetv2_baseline_otherdata.sh
# bash figure6/ace_run_steal_attack_TrainME_resnet20_all.sh
# bash figure6/ace_run_steal_attack_TrainME_mobilenetv2_all.sh

bash figure6/ace_run_steal_attack_TrainME_resnet18_cifar10.sh
# bash figure6/ace_run_resnet32_baseline.sh
# bash figure6/ace_run_resnet34_baseline.sh
bash figure6/ace_run_steal_attack_TrainME_resnet32_cifar10.sh
bash figure6/ace_run_steal_attack_TrainME_resnet34_cifar10.sh