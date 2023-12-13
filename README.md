# EMGAN: Early-Mix-GAN on Extracting Server-side Model in Split Federated Learning
This repository is official pytorch repository of two SFL MEA attack paper.

In the first work `Model Extraction Attacks on Split Federated Learning`, we start the investigation of MEA on SFL. Where we discover the interesting problem setting and propose five variants of ME attack which differs in the gradient usage as well as in the data assumptions. We make them all availble in the code without much cleanning - best performers `ganME` and `TrainME` (renamed as `naiveME`) are used as baseline in our second work and are included in the scripts.

The second work **(recently accepted by AAAI' 24)** builds on top of the first one, aiming to address the challange in perform MEA in training-from-scratch applications. In this work, we propose a strong MEA during the SFL training phase. 

## Abstract:
The proposed Early-Mix-GAN~(EMGAN) attack effectively exploits gradient queries regardless of data assumptions.
EMGAN adopts three key components to address the problem of inconsistent gradients. Specifically, it employs (i) Early-learner approach for better adaptability, (ii) Multi-GAN approach to introduce randomness in generator training to mitigate mode collapse, and (iii) ProperMix to effectively augment the limited amount of synthetic data for a better approximation of the target domain data distribution. EMGAN achieves excellent results in extracting server-side models.
With only 50 training samples, EMGAN successfully extracts a 5-layer server-side model of VGG-11 on CIFAR-10, with 7\% less accuracy than the target model. With zero training data, the extracted model achieves 81.3\% accuracy, which is significantly better than the 45.5\% accuracy of the model extracted by the SoTA method.

## Requirement:
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
wandb
thop
transformers

## Code:

* *SFL.py*: It implements the all utility functions of split learning and model extraction attacks during SFL (collection step)
  
* *attacks/model_extraction_attack.py*: It implements the model extraction attacks (as IP protection test).

* *attacks/model_inversion_attack.py*: It implements the model inversion attacks (as Data security test).

* *main.py*: Entry code to train a vanilla model. wandb registered

* *main_steal_offline.py*: Entry code to do static model extraction attacks (fine-tuning case).
  
* *main_steal_online.py*: Entry code to do model extraction attacks during model training (training-from-scratch case).

## Proof Of Concepts:

```
bash scripts/run_all_online.sh*
```

## Cite the work:

```
@article{li2023model,
  title={Model Extraction Attacks on Split Federated Learning},
  author={Li, Jingtao and Rakin, Adnan Siraj and Chen, Xing and Yang, Li and He, Zhezhi and Fan, Deliang and Chakrabarti, Chaitali},
  journal={arXiv preprint arXiv:2303.08581},
  year={2023}
}

@inproceedings{li2024emgan,
  title={EMGAN: Early-Mix-GAN on Extracting Server-side Model in Split Federated Learning},
  author={Li, Jingtao and Rakin, Adnan Siraj and Chen, Xing and Yang, Li and Fan, Deliang and Chakrabarti, Chaitali},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={x},
  number={x},
  year={2024}
}

```
