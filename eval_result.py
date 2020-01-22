import os
import torch
from torch import nn
import network_model_ResNet
import data_loader
import numpy as np

batch_size = 1

data_dir = '/nfs/masi/leeh43/CS8395_detection'
valid_data_dir = os.path.join(data_dir, 'validation')
label_dir = os.path.join(data_dir, 'labels')

## load label text ##
label_txt = os.path.join(label_dir, 'labels.txt')
with open(label_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]

## Training data list & Validataion data list##
valid_list = []
for item in content:
    image_name = item.split(' ')[0]
    if image_name in os.listdir(valid_data_dir):
        valid_list.append(item)

## Validation list ##
valid_file_list = []
valid_label_list = []
for data in valid_list:
    valid_file = os.path.join(valid_data_dir, data.split(' ')[0])
    valid_file_list.append(valid_file)
    valid_label = np.array([data.split(' ')[1], data.split(' ')[2]])
    valid_label_list.append(valid_label)

## Input into Pytorch Dataloader ##
valid_dict = {}
valid_dict['image_file'] = valid_file_list
valid_dict['image_label'] = valid_label_list

Valid_set = data_loader.pytorch_loader(valid_dict)
Valid_loader = torch.utils.data.DataLoader(Valid_set,batch_size=batch_size,shuffle=True,num_workers=1)


epoch = 80
model_dir = os.path.join(data_dir, 'resnet34_lr=0.0001_batch32', 'models')
pytorch_model = os.path.join(model_dir, 'model_epoch_00'+ str(epoch) + '.pth')
checkpoint = torch.load(pytorch_model)
model = network_model_ResNet.resnet34()

model.load_state_dict(checkpoint)
model.to(device='cuda')

loss_list = []
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(Valid_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        # obtain the outputs from the model
        outputs = model.forward(inputs)
        print(outputs)
        print(labels)
        criterion = nn.L1Loss()
        loss = criterion(outputs, labels)
        print('Loss = %.4f' % loss.data)
