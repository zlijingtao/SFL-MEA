# MIA Defense: Resistance-Ready: Attacker-Aware Training + NetworkBottleneck
This repository is official pytorch repository of Resistance-Ready -  for defending MIA in split learning.

## Requirement:
matplotlib

tqdm

tensorboard>1.15

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

thop (pip install)

## Code:

* *MIA_torch.py*: It implements the all utility functions of split learning and running MIA attacks.

* *main_MIA.py*: Entry code to train a defensive model/vanilla model.

* *main_test_MIA.py*: Entry code to resume a trained model and perform MIA attack.

## Proof Of Concepts:



## Cite the work:
```
@misc{li2021resistant,
      title={Resistant-Ready: xxx},
      author={Jingtao Li and others}, 
      year={2021},
      eprint={2012.02670},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
