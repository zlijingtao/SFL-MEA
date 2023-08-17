# EMGAN: early-mix-gan to steal IP of SFL
This repository is official pytorch repository of EMGAN

EMGAN, an advanced adaptive ME attack.
Checkout 04dcce44aa2 for the old version, including five basic ME attacks (many are deprecated in the current version)

## Roadmap:
[x] Propose five basic ME attacks
[x] Improve Gan-based data-free extraction attack
[x] Paper Writing

## Requirement:

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

wandb (pip install)

thop (pip install)

transformers (pip install)

## Code:

* *SFL.py*: It implements the all utility functions of split learning and model extraction attacks during SFL (collection step)
  
* *attacks/model_extraction_attack.py*: It implements the model extraction attacks (as IP protection test).

* *attacks/model_extraction_attack.py*: It implements the model inversion attacks (as Data security test).

* *main.py*: Entry code to train a vanilla model. wandb registered

* *main_steal_online.py*: Entry code to perform model extraction attacks during training.

## Proof Of Concepts:

* *bash scripts/run_attack.sh*

## Cite the work:
