# from datasets import load_dataset
import sys
sys.path.append("../")

from datasets import get_dataset
from transformers import ViTForImageClassification


# load cifar10 (only small portion for demonstration purposes) 
id2label = {0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'}

label2id = {label:id for id,label in id2label.items()}
print(label2id)
train_loader, test_loader, _ = get_dataset('cifar10', batch_size=128, actual_num_users=1, augmentation_option=True)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)


# id2label
# split up training into training + validation
# splits = train_ds.train_test_split(test_size=0.1)
# train_ds = splits['train']
# val_ds = splits['test']