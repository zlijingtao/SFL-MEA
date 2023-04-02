# SFL IP protection: a model split approach to protect IP of FL
This repository is official pytorch repository of SFL IP protection

## Requirement:

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

wandb (pip install)

thop (pip install)

transformers (pip install)

## Code:

* *SFL.py*: It implements the all utility functions of split learning and model extraction attacks during SFL (collection step)
  
* *attacks/model_extraction_attack.py*: It implements the model extraction attacks (final step).

* *main.py*: Entry code to train a vanilla model. wandb registered

* *main_vit.py*: Entry code to finetune a ViT model. wandb registered

* *main_steal_offline.py*: Entry code to do post-analysis of all model extraction attacks.
* 
* *main_steal_online.py*: Entry code to do post-analysis of all model extraction attacks.

## Proof Of Concepts:



## Cite the work:
