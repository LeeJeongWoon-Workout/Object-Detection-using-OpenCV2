#Google Colab 환경 기준

!mkdir /content/data  #linux source code that makes directory in content
!wget -O ./data/beatles01.jpg https://raw.githubusercontent.com/chulminkw/DLCV/master/data/image/beatles01.jpg # linux source code that brings data and rename the file-name


import cv2  #OpenCV 파일 import
import os
import matplotlib.pyplot as plt
%matplotlib inline

img=cv2.imread('./data/beatles01.jpg') #cv2를 통해 이미지를읽어온다
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #cv2가 받아들이는 사진은 BGR 이기 때문에 RGB로 바꾸기 위해 cv2의 내장함수 cvtColor를 사용한다.
plt.imshow(img_rgb) #plt 를 통해 numpy array 자료를 이미지로 바꾸는 과정
plt.figure(figsize=(8,8))
plt.show()

!mkdir ./pretrained #pretrained directory made
!wget -O ./pretrained/faster_rcnn_resnet50_coco_2018_01_28.tar.gz http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz #텐서플로우로 미리 학습받은 모델의 backbone resnet weights 받는다.
!wget -O ./pretrained/config_graph.pbtxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/faster_rcnn_resnet50_coco_2018_01_28.pbtxt #받은 weight를 cv2가 해석할 수 있도록 config를 받는다.
  #tensorflow로 작성된 resnet backbone으로 이루어진  faster RCNN 모델을  가져온다.
  #openCV 가져온 모델을 해석할 수 있도록 config_graph 가이드를 가져온다.
  
  
!tar -xvf ./pretrained/faster*.tar.gz -C ./pretrained  #압축풀기 여기에 있는 압축파일을./pretrained에 푼다.
cv_net = cv2.dnn.readNetFromTensorflow('./pretrained/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb', 
                                     './pretrained/config_graph.pbtxt') #weight 정보 가이드를 기반으로 텐서플로우로 짜진 DNN net을 cv2 내장 함수로 만든다.
