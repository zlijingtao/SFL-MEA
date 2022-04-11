# SFL IP protection: a model split approach to protect IP of FL
This repository is official pytorch repository of SFL IP protection

## Requirement:
matplotlib

tqdm

tensorboard>1.15

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

thop (pip install)

## Code:

* *MIA_torch.py*: It implements the all utility functions of split learning and running model extraction attacks during SFL

* *MIA_torch.py - self.steal_attack*: It implements all model extraction attacks as post analysis on a trained SFL model.

* *main_MIA.py*: Entry code to train a defensive model/vanilla model.

* *main_test_MIA.py*: Entry code to resume a trained model.

* *main_model_steal.py*: Entry code to do post-analysis of all model extraction attacks.

## Proof Of Concepts:



## Cite the work:
