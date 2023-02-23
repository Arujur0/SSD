import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]

    pred_box = torch.reshape(pred_box,(-1, 4))
    pred_confidence = torch.reshape(pred_confidence,(-1, 4))
    ann_confidence = torch.reshape(ann_confidence,(-1, 4))
    ann_box = torch.reshape(ann_box, (-1, 4))


    isobj = torch.where(ann_confidence[:,-1] == 0)
    noobj = torch.where(ann_confidence[:, -1] == 1)
    conf_loss = F.cross_entropy(pred_confidence[isobj], ann_confidence[isobj]) + 3 * F.cross_entropy(pred_confidence[noobj], ann_confidence[noobj])
    box_loss = F.smooth_l1_loss(pred_box[isobj], ann_box[isobj])
    #print("Loss with softmax: ", conf_loss + box_loss)
    return conf_loss + box_loss

class SSD(nn.Module):
    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.SSD = nn.Sequential(nn.Conv2d(3, 64, 3, 2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                nn.Conv2d(64, 128, 3, 2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                nn.Conv2d(128, 256, 3, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 256, 3, 1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                nn.Conv2d(256, 512, 3, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                nn.Conv2d(512, 512, 3, 1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                nn.Conv2d(512, 512, 3, 1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                nn.Conv2d(512, 256, 3, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(256,256,1,1, padding=0), nn.BatchNorm2d(256), nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(256,256,1,1, padding=0), nn.BatchNorm2d(256), nn.ReLU(),
                            nn.Conv2d(256, 256, 3, 1, padding=0), nn.BatchNorm2d(256), nn.ReLU())
        self.conv31 = nn.Sequential(nn.Conv2d(256,256,1,1, padding=0), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, 1, padding=0), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Conv2d(256, 16, 3, 1, padding=1)
        self.conv7 = nn.Conv2d(256, 16, 1, 1, padding = 0)
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        x = self.SSD(x)
        bboxes_x = self.conv6(x)
        conf_x = self.conv6(x)
        y = self.conv5(x)
        bboxes_y = self.conv6(y)
        conf_y = self.conv6(y)
        z = self.conv3(y)
        bboxes_z = self.conv6(z)
        conf_z = self.conv6(z)
        w = self.conv31(z)
        bboxes_w = self.conv6(w)
        conf_w = self.conv6(w)

        bboxes_x =torch.reshape(bboxes_x, (-1, 16, 100))
        conf_x = torch.reshape(conf_x, (-1,16, 100))
        bboxes_y =torch.reshape(bboxes_y, (-1, 16, 25))
        conf_y = torch.reshape(conf_y, (-1,16, 25))
        bboxes_z =torch.reshape(bboxes_z, (-1, 16, 9))
        conf_z = torch.reshape(conf_z, (-1,16, 9))
        bboxes_w =torch.reshape(bboxes_w, (-1, 16, 1))
        conf_w = torch.reshape(conf_w, (-1,16, 1))

        bboxes = torch.cat((bboxes_x, bboxes_y, bboxes_z,bboxes_w), axis = 2)
        confidence = torch.cat((conf_x, conf_y, conf_z, conf_w), axis = 2)

        bboxes = torch.permute(bboxes,(0, 2, 1))
        confidence = torch.permute(confidence,(0, 2, 1))


        bboxes = torch.reshape(bboxes, (-1, 540, 4))
        confidence = torch.reshape(confidence, (-1, 540, 4))

        
        return confidence,bboxes










