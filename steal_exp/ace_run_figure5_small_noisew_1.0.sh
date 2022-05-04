#!/bin/bash
cd "$(dirname "$0")"

bash figure5_gan/ace_run_vgg11_cifar10_train_gan.sh
bash figure5_gan/ace_run_vgg11_cifar10_train_gan_resume.sh