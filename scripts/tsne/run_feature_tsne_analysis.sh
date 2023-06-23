#!/bin/bash
cd "$(dirname "$0")"

bash run_ganassistME_vgg11bn_cifar10_baseline_running_margin.sh
bash run_ganassistME_vgg11bn_cifar10_proposed_running_margin.sh
bash run_ganassistME_vgg11bn_cifar10_sameclass_running_margin.sh
bash run_ganassistME_vgg11bn_cifar10_undermix_running_margin.sh

cd ../../

python tools/tsne_feature_analysis.py