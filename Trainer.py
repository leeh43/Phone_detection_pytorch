import datetime
import math
import os
import os.path as osp
import shutil
from torch import nn
# import fcn
import numpy as np
from sklearn.metrics import roc_auc_score
import pytz
import scipy.misc
import scipy.io as sio
import nibabel as nib
from scipy.spatial import distance
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import skimage
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Regression_trainer(object):
    def __init__(self, cuda, model, optimizer=None,
                train_loader=None,test_loader=None,lmk_num=None,
                train_root_dir=None,out=None, output_model=None, 
                test_data=None, max_epoch=None, batch_size=None,
                size_average=False, interval_validate=None,
                compete = False,onlyEval=False):
        if torch.cuda.is_available():
            self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.interval_validate = interval_validate

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        self.train_root_dir = train_root_dir
        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.lmk_num = lmk_num
        self.output_model = output_model
        self.test_data = test_data

        self.max_epoch = max_epoch
        self.epoch = max_epoch
        self.iteration = 0
        self.best_mean_iu = 0
        self.batch_size = batch_size

    def train(self):
        self.model.train()
        out = osp.join(self.out, 'visualization')
        mkdir(out)
        log_file = osp.join(out, 'training_loss.txt')
        fv = open(log_file, 'a')
        torch.set_num_threads(1)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred = self.model(data)
            self.optim.zero_grad()

            criterion = nn.MSELoss()
            loss = criterion(pred, target)
            loss.backward()
            self.optim.step()

            print('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))

            fv.write('epoch=%d, batch_idx=%d, loss=%.4f \n' % (self.epoch, batch_idx, loss.data[0]))

    def train_epoch(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            out = osp.join(self.out, 'models')
            mkdir(out)

            model_pth = '%s/model_epoch_%04d.pth' % (out, epoch)

            if os.path.exists(model_pth):
                if self.cuda:
                    self.model.load_state_dict(torch.load(model_pth))
                else:
                    # self.model.load_state_dict(torch.load(model_pth))
                    self.model.load_state_dict(
                        torch.load(model_pth, map_location=lambda storage, location: storage))
                # if epoch % 5 == 0:
                # self.validate()
            else:
                self.train()
                # if epoch % 5 == 0:
                #     self.validate()
                torch.save(self.model.state_dict(), model_pth)
                if self.cuda:
                    self.model.load_state_dict(torch.load(model_pth))
                loss_list = []
                
                with torch.no_grad():
                    for idx, (inputs, labels) in enumerate(self.test_loader):
                        if self.cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()
                # obtain the outputs from the model
                        outputs = self.model.forward(inputs)
                        criterion = nn.MSELoss()
                        loss = criterion(outputs, labels)
                        loss_list.append(loss)
            
                loss_epoch = sum(loss_list) / len(loss_list)  
                print('Validation_loss=%.4f \n' % (loss_epoch))
                out = os.path.join(self.out, 'visualization')
                log_file = os.path.join(out, 'validation_loss.txt')
                fv = open(log_file, 'a')
                fv.write('loss=%.4f \n' % (loss_epoch.data[0]))


    def calc_accuracy(self, output_model, test_data):
        pytorch_model = self.output_model
        checkpoint = torch.load(pytorch_model)
        self.model.load_state_dict(checkpoint)
        self.model.to(device='cuda')    

        loss_list = []
        with torch.no_grad():
            for idx, (inputs, labels, subject_name) in enumerate(self.test_loader):
                if self.cuda:
                   inputs, labels = inputs.cuda(), labels.cuda()
                # obtain the outputs from the model
                outputs = self.model.forward(inputs)
                criterion = nn.MSELoss()
                loss = criterion(outputs, labels)
                loss_list.append(loss)
        loss_epoch = sum(loss_list) / len(loss_list)  
        print('Testing_loss=%.4f \n' % (loss_epoch))
        out = os.path.join(self.out, 'visualization')
        log_file = os.path.join(out, 'testing_loss.txt')
        fv = open(log_file, 'a')
        fv.write('loss=%.4f \n' % (loss_epoch.data[0]))
            # check the 
                # if idx == 0:
                # print(outputs) #the predicted class 
                # print(labels)
