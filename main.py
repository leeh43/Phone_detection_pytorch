import os
import torch
from torch import nn
import torch.utils.data.dataloader as dataloader
import data_loader
import network_model_ResNet
import Trainer
import numpy as np
import pandas as pd
import os
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

## Training parameters ##
batch_size = 64
learning_rate = 0.0001
epoch_num = 200


data_dir = '/nfs/masi/leeh43/CS8395_detection'
train_data_dir = os.path.join(data_dir, 'train')
valid_data_dir = os.path.join(data_dir, 'validation')
label_dir = os.path.join(data_dir, 'labels')

## load label text ##
label_txt = os.path.join(label_dir, 'labels.txt')
with open(label_txt) as f:
    content = f.readlines()
content = [x.strip() for x in content]

## Training data list & Validataion data list##
train_list = []
valid_list = []
for item in content:
    image_name = item.split(' ')[0]
    if image_name in os.listdir(train_data_dir):
        train_list.append(item)
    if image_name in os.listdir(valid_data_dir):
        valid_list.append(item)

## Train list ##
train_file_list = []
train_label_list = []
for data in train_list:
    train_file = os.path.join(train_data_dir, data.split(' ')[0])
    train_file_list.append(train_file)
    train_label = np.array([data.split(' ')[1], data.split(' ')[2]])
    train_label_list.append(train_label)

## Validation list ##
valid_file_list = []
valid_label_list = []
for data in valid_list:
    valid_file = os.path.join(valid_data_dir, data.split(' ')[0])
    valid_file_list.append(valid_file)
    valid_label = np.array([data.split(' ')[1], data.split(' ')[2]])
    valid_label_list.append(valid_label)

## Input into Pytorch Dataloader ##
train_dict = {}
train_dict['image_file'] = train_file_list
train_dict['image_label'] = train_label_list

valid_dict = {}
valid_dict['image_file'] = valid_file_list
valid_dict['image_label'] = valid_label_list

Train_set = data_loader.pytorch_loader(train_dict)
Train_loader = torch.utils.data.DataLoader(Train_set,batch_size=batch_size,shuffle=True,num_workers=1)
Valid_set = data_loader.pytorch_loader(valid_dict)
Valid_loader = torch.utils.data.DataLoader(Valid_set,batch_size=batch_size,shuffle=True,num_workers=1)

## Load deep neural network model ##
model = network_model_ResNet.resnet50()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Output model directory ##
output_dir = os.path.join('/nfs/masi/leeh43/CS8395_detection')
out = os.path.join(output_dir, 'resnet50_lr=0.0001_batch64')
if os.path.exists(out) == False:
    os.mkdir(out)

## Load cuda for training with GPU ##
cuda = torch.cuda.is_available()
torch.manual_seed(1)
if cuda:
	torch.cuda.manual_seed(1)
	model = model.cuda()

trainer = Trainer.Regression_trainer(cuda=cuda,
                             model=model,
                             optimizer=optimizer,
                             train_loader=Train_loader,
                             test_loader=Valid_loader,
                             train_root_dir=None,
                             out=out,
                             output_model=None,
                             test_data=Valid_loader,
                             max_epoch=epoch_num,
                             batch_size=batch_size)

print("==start training==")


start_iteration = 0
start_epoch = 0

trainer.epoch = start_epoch
trainer.iteration = start_iteration
trainer.train_epoch()

