# from datasets import load_dataset
import sys
sys.path.append("../")

from datasets import get_dataset
from transformers import ViTForImageClassification
from models.vit_wrapper import vit_split_model_wrapper
import torch
import torchvision.transforms as transforms

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

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)

cutlayer = 10
num_client = 2
freeze_front_layer = 8
train_loader, test_loader, _ = get_dataset('cifar10', batch_size=128, actual_num_users=num_client, augmentation_option=True)

split_model = vit_split_model_wrapper(model, cutlayer, num_client, freeze_front_layer)
split_model.cuda()
# exit()

dl_transform = dl_transforms = torch.nn.Sequential(
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    )
# print(split_model.parameters())
optimizer = torch.optim.Adam(split_model.cloud.parameters(), lr=0.001)

local_optimizer_list = [torch.optim.Adam(split_model.local_list[0].parameters(), lr=0.001)]
for client_id in range(1, num_client):
    local_optimizer_list.append(torch.optim.Adam(split_model.local_list[client_id].parameters(), lr=0.001))
    local_optimizer_list[client_id].zero_grad()

criterion = torch.nn.CrossEntropyLoss()

count = 0
optimizer.zero_grad()

for client_id in range(num_client):
    split_model.switch_model(client_id)
    for image, label in train_loader[client_id]:
        print(f"iter-{count}")
        count += 1
        image = dl_transform(image)
        image = image.cuda()
        label = label.cuda()

        logits = split_model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        local_optimizer_list[client_id].step()