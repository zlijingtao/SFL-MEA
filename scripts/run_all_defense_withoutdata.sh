#!/bin/bash
cd "$(dirname "$0")"
# bash defense/run_ganasistME_gradient_noise.sh
# bash defense/run_ganasistME_reduce_grad_freq.sh
# bash defense/run_ganasistME_label_smooth.sh
# bash defense/run_ganasistME_l1.sh
bash defense/run_ganME_gradient_noise.sh
bash defense/run_ganME_l1.sh

