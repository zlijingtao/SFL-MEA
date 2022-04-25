#!/bin/bash
cd "$(dirname "$0")"

bash figure5/ace_run_vgg11_cifar10_train_GM.sh
bash figure5/ace_run_vgg11_cifar10_train_GM_resume.sh
