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
import numpy as np
import os
import cv2
from PIL import Image
#generate default bounding boxes

def gen_box(k, small, large, box, layers, i, j):
    sscale = [small[k], large[k], large[k] * np.sqrt(2), large[k] / np.sqrt(2)]
    lscale = [small[k], large[k], large[k] / np.sqrt(2), large[k] * np.sqrt(2)]
    xcenter = ((j + 0.5)/layers[k])
    ycenter = ((i + 0.5)/layers[k])
    for l in range(4):
        box[l, 0] = xcenter
        box[l, 1] = ycenter
        box[l, 2] = sscale[l]
        box[l, 3] = lscale[l]
        box[l, 4] = xcenter - 0.5 * sscale[l] if xcenter - 0.5 * sscale[l] > 0 else 0
        box[l, 5] = ycenter - 0.5 * lscale[l] if ycenter - 0.5 * lscale[l] > 0 else 0
        box[l, 6] = xcenter + 0.5 * sscale[l] if xcenter + 0.5 * sscale[l] <= 1 else 1
        box[l, 7] = ycenter + 0.5 * lscale[l] if ycenter + 0.5 * lscale[l] <= 1 else 1
    return box
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    box = np.zeros((135, 4, 8))
    k = 0
    layerno = 0

    while(k<4):
        for i in range(layers[k]):
            for j in range(layers[k]):
                box[layerno, :, :] = gen_box(k, small_scale, large_scale, box[layerno], layers, i, j)
                layerno+=1
        k+=1

    boxes = box.reshape(540, 8)
    
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max, x):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    flag = 0
    ious_true = ious>threshold
    #TODO:
    for j in range(len(ious_true)):
        if ious_true[j]:
            flag = 1
            if cat_id == 0:
                ann_confidence[j][0] = 1
                ann_confidence[j][1] = 0
                ann_confidence[j][2] = 0
                ann_confidence[j][3] = 0
            elif cat_id == 1:
                ann_confidence[j][0] = 0
                ann_confidence[j][1] = 1
                ann_confidence[j][2] = 0
                ann_confidence[j][3] = 0
            elif cat_id == 2:
                ann_confidence[j][0] = 0
                ann_confidence[j][1] = 0
                ann_confidence[j][2] = 1
                ann_confidence[j][3] = 0
            tx = (x[0] - boxs_default[j][0]) / boxs_default[j][2]
            ty = (x[1] - boxs_default[j][1]) / boxs_default[j][3]
            tw = np.log(x[2]/ boxs_default[j][2])
            th = np.log(x[3] / boxs_default[j][3])
            ann_box[j][0] = tx
            ann_box[j][1] = ty
            ann_box[j][2] = tw
            ann_box[j][3] = th
    if flag == 0:
        ious_true = np.argmax(ious)
        if cat_id == 0:
            ann_confidence[ious_true][0] = 1
            ann_confidence[ious_true][1] = 0
            ann_confidence[ious_true][2] = 0
            ann_confidence[ious_true][3] = 0
        elif cat_id == 1:
            ann_confidence[ious_true][0] = 0
            ann_confidence[ious_true][1] = 1
            ann_confidence[ious_true][2] = 0
            ann_confidence[ious_true][3] = 0
        elif cat_id == 2:
            ann_confidence[ious_true][0] = 0
            ann_confidence[ious_true][1] = 0
            ann_confidence[ious_true][2] = 1
            ann_confidence[ious_true][3] = 0
        tx = (x[0] - boxs_default[ious_true][0]) / boxs_default[ious_true][2]
        ty = (x[1] - boxs_default[ious_true][1]) / boxs_default[ious_true][3]
        tw = np.log(x[2]/ boxs_default[ious_true][2])
        th = np.log(x[3] / boxs_default[ious_true][3])
        ann_box[ious_true][0] = tx
        ann_box[ious_true][1] = ty
        ann_box[ious_true][2] = tw
        ann_box[ious_true][3] = th
    return ann_box, ann_confidence
    #TODO:



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320, test = False):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        self.test = test
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        if train:
            self.img_names = self.img_names[:int(0.9 * len(self.img_names))]
        elif train and not test:
            self.img_names = self.img_names[int(0.9 * len(self.img_names)):]
        elif test and not train:
            self.img_names = self.img_names
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        #print(index)
        #TODO:
        img = Image.open(img_name)
        #i1 = transforms.ToTensor()
        im1_w, im1_h = img.size
        im = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor()])
        image = im(img)
        if image.shape[0] != 3:
            image = torch.cat((image, image, image), axis = 0)
        #DONT FORGET TO RESIZE THE IMAGE
        if self.test:
            return image
        with open(ann_name) as f:
            lines = f.readlines()
        for i in lines:
            line = i.split()
            x = np.array(line[1:], dtype=float)
            x_min = x[0] 
            y_min = x[1]
            x[2] = x[2] / im1_w
            x[3] = x[3] / im1_h
            x_min = x_min / im1_w
            y_min = y_min / im1_h
            x_center = x_min + 0.5 * x[2]
            #x_center = x_center/im1.shape[2]
            x[0] = x_center
            y_center = y_min + 0.5 * x[3]
            #y_center = y_center/im1.shape[1]
            x[1] = y_center
            x_max = x_min + x[2]
            y_max = y_min + x[3]
            #print(x)
            ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold, int(line[0]), x_min, y_min, x_max, y_max, x)
    
        return image, ann_box, ann_confidence
