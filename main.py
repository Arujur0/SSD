import argparse
import os
import numpy as np
import time
import cv2
import wandb
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test

def calc(box_, boxs_default):
    x_center = boxs_default[2] * box_[0] + boxs_default[0]
    y_center = boxs_default[3] * box_[1] + boxs_default[1]

    w = boxs_default[2] * np.exp(box_[2])
    h = boxs_default[3] * np.exp(box_[3])
    return [x_center, y_center, w, h]

class_num = 4 #cat dog person background

num_epochs = 1
batch_size = 48


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])

network = SSD(class_num)
network.cuda()
cudnn.benchmark = True
load = True
if load:
  print('loading model')
  network.load_state_dict(torch.load('network11.pth'))

if not args.test:
    dataset = COCO("A1\workspace\data\\train\images\\", "A1\workspace\data\\train\\annotations\\", class_num, boxs_default, train = True, image_size=320)
    dataset_test = COCO("A1\workspace\data\\train\images\\", "A1\workspace\data\\train\\annotations\\", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.AdamW(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    for epoch in range(num_epochs):
        #TRAINING
        network.train()
        print("Epoch number : ", epoch)
        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1  

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        #wandb.log({"train loss": avg_loss/avg_count})
        
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #VALIDATION
        network.eval()
        val_loss = 0
        val_count = 0
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()

            val_loss += loss_net.data
            val_count += 1


            break
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        print('[%d] time: %f validation loss: %f' % (epoch, time.time()-start_time, val_loss/val_count))
        visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        if epoch%10==9:
            #save last network  
            print('saving net...')
            torch.save(network.state_dict(), 'network11.pth')


else:
    #TEST
    dataset_test = COCO("A1\workspace\data\\train\images\\", "A1\workspace\data\\train\\annotations\\", class_num, boxs_default, train = False, image_size=320, test = True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network11.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        #images_, ann_box_, ann_confidence_ = data
        images_ = data
        images = images_.cuda()
        ann_confidence_ = np.zeros((540, 4))
        ann_box_ = np.zeros((540,4))
        # ann_box = ann_box_.cuda()
        # ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default) 
        visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_, ann_box_, images_[0].numpy(), boxs_default, test = True)    
        cv2.waitKey(1000)


