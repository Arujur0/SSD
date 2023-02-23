# SSD
## Introduction 
This is a foundational implementation of SSD Object Detection. SSD divides the image into grids and outputs bounding boxes in each cell

## Architecture
It is based on [SSD: Single Shot MultiBox Detector] (https://arxiv.org/pdf/1512.02325.pdf). The architecture consists of a series of Convolutional layers, the output of the convolutional layers represents the different types of grids possible. The grids are 10 x 10, 5x5, 3x3 and 1x1. There are two paths for the predictions; one for the bounding boxes and one for the class confidences. The architecture is given below:
![SSD Multibox Detector Architecture](arch.png)

## Implementation
The neural network was trained for 250 epochs with a learning rate of 1e-4, using the Adam optimizer. The number of epochs, learning rate and batch size can be modified in the main.py file.In addition to the basic implementation Non-Maximum Suppression was implemented to make the model better accurate.
![Non Maximum Suppression](nms.png)

##Results
To run the train loop simply run 
  python main.py
To run the test loop simply add the --test argument to the train loop command.
  python main.py --test
  
The results of the model after implementing Non-Maximum Suppression are fairly good.
![Image 1](imag1.png)
![Image 2](imag2.png)

The image to the left is the model's prediction while the image to the right is the ground truth bounding box.
