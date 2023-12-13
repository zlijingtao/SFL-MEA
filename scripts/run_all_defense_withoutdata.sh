#!/bin/bash
cd "$(dirname "$0")"
bash defense/run_ganME_reduce_grad_freq.sh
bash defense/run_ganME_gradient_noise.sh
bash defense/run_ganME_l1.sh

