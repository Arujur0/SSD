import numpy as np
import cv2
from dataset import iou
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import softmax
import torch
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, test = False):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    #print(pred_confidence)
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ *= 255
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]

    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):   
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has hith confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                tx = boxs_default[i][2]*ann_box[i][0] + boxs_default[i][0]
                ty = boxs_default[i][3]*ann_box[i][1] + boxs_default[i][1]
                tw = boxs_default[i][2]*np.exp(ann_box[i][2])
                th = boxs_default[i][3]*np.exp(ann_box[i][3])

                start = (int((tx-(tw * 0.5)) * 320), int((ty-(th * 0.5))* 320))
                end = (int((tx+(tw * 0.5))* 320), int((ty+(th * 0.5)) * 320))
                #start = ann_box[]
                #end = ann_box[]
                color = colors[j]
                thickness = 2
                cv2.rectangle(image1,start, end, color, thickness)

                start_point = (int((boxs_default[i][0]-(boxs_default[i][2]/2))* 320), int((boxs_default[i][1]-(boxs_default[i][3]/2))* 320))

                end_point = (int((boxs_default[i][0]+(boxs_default[i][2]/2)) * 320), int((boxs_default[i][1]+(boxs_default[i][3]/2))* 320))
                #print(start_point, end_point)
                cv2.rectangle(image2, start_point, end_point, color, thickness)
    
    #pred
    
    pred_confidence = softmax(pred_confidence, axis=1)
    for i in range(len(pred_confidence)):
        #print(pred_confidence[i])
        for j in range(class_num):
            if pred_confidence[i,j]>0.3:
                #TODO:
                #print("Predicted Confidence", pred_confidence[i, j])
                tx = boxs_default[i][2]*pred_box[i][0] + boxs_default[i][0]
                ty = boxs_default[i][3]*pred_box[i][1] + boxs_default[i][1]
                tw = boxs_default[i][2]*np.exp(pred_box[i][2])
                th = boxs_default[i][3]*np.exp(pred_box[i][3])

                start = (int((tx-(tw * 0.5)) * 320), int((ty-(th * 0.5))* 320))
                end = (int((tx+(tw * 0.5))* 320), int((ty+(th * 0.5)) * 320))
                
                start_point = (int((boxs_default[i][0]-(boxs_default[i][2]/2)) * 320), int((boxs_default[i][1]-(boxs_default[i][3]/2))* 320))
                end_point = (int((boxs_default[i][0]+(boxs_default[i][2]/2)) * 320),int((boxs_default[i][1]+(boxs_default[i][3]/2))* 320))
                # print(start, end, start_point, end_point)
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start, end, color, thickness)

                cv2.rectangle(image4, start_point, end_point, color, thickness)

                write = [j, tx, ty, tw,th]

                file = open("results.txt", "a")
                file.writelines(str(write) + "\n")

                file.close()
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if windowname == "val":
        cv2.imwrite("imageval.jpg", image)
    if windowname == "test":
        cv2.imwrite("imagetest.jpg", image)
    cv2.imwrite("imagetrain.jpg", image)
   

def calcbox(box_, boxs_default):
    x_center = boxs_default[2] * box_[0] + boxs_default[0]
    y_center = boxs_default[3] * box_[1] + boxs_default[1]

    w = boxs_default[2] * np.exp(box_[2])
    h = boxs_default[3] * np.exp(box_[3])
    xmin = x_center - 0.5 * w
    ymin = y_center - 0.5 * h
    xmax = xmin + w
    ymax = ymin + h
    box = [x_center, y_center, w, h, xmin, ymin, xmax, ymax]
    return box

def calcmins(predicted):
    return predicted[0] - (predicted[2] / 2), predicted[1] - (predicted[3] / 2), predicted[0] + (predicted[2] / 2), predicted[1] + (predicted[3] / 2)


def iou_calc(boxs_default, x_min,y_min,x_max,y_max):
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

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.2):
    #confidence_ = softmax(confidence_, axis=1)
    # conf = np.argsort(g, axis = 1)
    prior = box_.copy()
    predicted = np.empty((540,8),dtype=float)
    for i in range(540):
        pred = calcbox(box_[i], boxs_default[i])
        predicted[i] = pred
    all_conf = confidence_.copy()
    b = []
    max_conf = 0
    index = 0
    count = 1
    while count < 20:
        for j in range(len(all_conf)):
            if j not in b:
                if np.max(all_conf[j, :3]) > max_conf:
                    max_conf = np.max(all_conf[j,:3])
                    index = j
        if max_conf < threshold:
            #count = 0
            return all_conf, prior
        a = np.delete(predicted, index, 0)
        b.append(index)
        ious = iou_calc(a, predicted[index][4], predicted[index][5], predicted[index][6], predicted[index][7])
        iou_t = np.where(ious > 0.05)[0]
        for i in range(len(iou_t)):
            all_conf[iou_t[i]] = [0, 0, 0, 1]
        count+=1
    return all_conf, prior
    

def generate_mAP(pred_box, ann_box, ann_conf, pred_conf, boxs_default, threshold = 0.3):
    mAP = 0
    classes = [[1,0,0,0], [0,1,0,0],[0,0,1,0]]
    precision = []
    recall = []
    for i in range(3):
        # correct_gt = []
        # correct_pred = []
        # indexs_pred = []
        # indexs_gt = []
        print(pred_conf.shape)
        ann_conf = np.reshape(ann_conf, (540, 4))
        ann_box = np.reshape(ann_box, (540, 4))
        indexs_gt = np.where(ann_conf[:, i] == 1)
        print(len(indexs_gt))
        indexs_pred = np.where(pred_conf[:,i] > 0.4)[0]
        correct_gt = ann_box[indexs_gt]
        correct_pred = pred_box[indexs_pred]
        # for j in range(len(ann_conf)):
        #     if np.all(ann_conf[j] == classes[i]):
        #         correct_gt.append(ann_box[j])
        #         indexs_gt.append(j)
                    
        # for j in range(len(pred_conf)):
        #     if np.argmax(pred_conf[j]) == i and np.max(pred_conf[j]) > threshold:
        #         correct_pred.append(pred_box[j])
        #         indexs_pred.append(j)

        TP = np.zeros(len(correct_pred))
        FP = np.zeros(len(correct_pred))
        predicted = np.empty((540, 8))
        annotated = np.empty((540, 8))
        for p in range(len(correct_pred)):
            predicted[p] = calcbox(correct_pred[p], boxs_default[indexs_pred[p]])
        total_boxes = len(correct_gt)
        for a in range(total_boxes):
            annotated[a] = calcbox(correct_gt[a], boxs_default[indexs_gt[a]])
        for ind, pred in enumerate(annotated):
            x_min, y_min, x_max, y_max = calcmins(pred)
            ious = iou_calc(predicted, x_min,  y_min, x_max, y_max)
            best_iou = np.max(ious)
            best_index = np.argmax(ious)
            print(best_iou)
            if best_iou > threshold:
                TP[ind] = 1
            else:
                FP[ind] = 1

        cumulative_TP = np.cumsum(TP)
        cumulative_FP = np.cumsum(FP)
        recall.append(cumulative_TP / np.max(total_boxes, 1e-8))
        precision.append(cumulative_TP / cumulative_TP + cumulative_FP + 1e-8)
    
    return np.sum(precision)/3, np.sum(recall)/3
    #TODO: Generate mAP








