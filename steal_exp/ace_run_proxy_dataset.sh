#!/bin/bash
cd "$(dirname "$0")"
bash proxy_dataset/ace_run_vgg11_baseline.sh
bash proxy_dataset/ace_run_steal_attack_gpu0.sh