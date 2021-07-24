!mkdir /content/data  #linux source code that makes directory in content
!wget -O ./data/beatles01.jpg https://raw.githubusercontent.com/chulminkw/DLCV/master/data/image/beatles01.jpg # linux source code that brings data and rename the file-name


import cv2
import os
import matplotlib.pyplot as plt
%matplotlib inline

img=cv2.imread('./data/beatles01.jpg')
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.figure(figsize=(8,8))
plt.show()

!mkdir ./pretrained #pretrained directory made
!wget -O ./pretrained/faster_rcnn_resnet50_coco_2018_01_28.tar.gz http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz #텐서플로우로 미리 학습받은 모델의 backbone resnet weights 받는다.
!wget -O ./pretrained/config_graph.pbtxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt #받은 weight를 cv2가 해석할 수 있도록 config를 받는다.
  
  
  
