#!/bin/bash
cd "$(dirname "$0")"
bash baseline/ace_run_vgg11_baseline.sh
bash baseline/ace_run_resnet20_baseline.sh
bash baseline/ace_run_mobilenetv2_baseline.sh
bash baseline/ace_run_resnet18_baseline.sh