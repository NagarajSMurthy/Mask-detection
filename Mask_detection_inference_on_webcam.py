# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:59:10 2020

@author: nagaraj
"""

import numpy as np
import cv2

import torch

import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
from IPython import display

from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torchvision

# path to the pytorch model
#model_path = '/pytorch_models/'

# path to the lightning model
model_path = '/Lightning_models/model_epoch=02-val_loss=0.09.ckpt'

class MobileFaceMask(nn.Module):
    
    def __init__(self):
        super(MobileFaceMask,self).__init__()
        
        #self.resnet = torchvision.models.resnet18(pretrained=True) 
        #self.resnet.fc = nn.Linear(512,128)
        self.dropout = nn.Dropout(p=0.3)
        self.network = torchvision.models.mobilenet_v2(pretrained=True)
        #child_counter = 0
        for child in self.network.children():
            for param in child.parameters():
                param.requires_grad = False
        self.network.classifier = torch.nn.Sequential(nn.Linear(1280,256),
                                              nn.Tanh(),
                                              nn.Dropout(p=0.25),
                                              nn.Linear(256,2),
                                              nn.LogSoftmax())
        
    def forward(self,x):

        x = self.network(x)

        return x

'''class MobileFaceMask(nn.Module):
    
    def __init__(self):
        super(MobileFaceMask,self).__init__()
        
        #self.resnet = torchvision.models.resnet18(pretrained=True) 
        #self.resnet.fc = nn.Linear(512,128)
        self.dropout = nn.Dropout(p=0.3)
        self.network = torchvision.models.mobilenet_v2(pretrained=True)
        child_counter = 0
        for child in self.network.children():
            for param in child.parameters():
                param.requires_grad = False
        self.network.classifier = torch.nn.Sequential(nn.Linear(1280,512),
                                              nn.Tanh(),
                                              nn.Dropout(p=0.25),
                                              nn.Linear(512,128),
                                              nn.Tanh(),
                                              nn.Dropout(p=0.25),
                                              nn.Linear(128,2),
                                              nn.LogSoftmax())
        
    def forward(self,x):

        x = self.network(x)

        return x'''
    
model = MobileFaceMask()
#model = torch.load(model_path,map_location='cpu')

# Loading lightning models saved as a checkpoint file
state_dict = torch.load(model_path)
model.load_state_dict(state_dict['state_dict'])

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((200,200)),
                           transforms.ToTensor()])

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

labels = ['Mask','No Mask']
labelColor = [(10, 255, 0),(10, 0, 255)]

cap = cv2.VideoCapture(0)

# MTCNN for detecting the presence of faces 
mtcnn = MTCNN(keep_all=True,device = device)

model.to(device)
model.eval()
while True:
    ret, frame = cap.read()
    if ret==False:
        pass
    
    img_ = frame.copy()
    boxes, _ = mtcnn.detect(img_)
    # Using PIL to draw boxes
    '''frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)'''
    '''
    try:
        for x1,y1,x2,y2 in boxes:
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            roi = img_[int(y1):int(y2) , int(x1):int(x2)]
    except TypeError as e:
        pass'''
    
    try:
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i]
            x1 , y1 = max(x1,0), max(y1,0)
            frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            face = img_[int(y1-30):int(y2+30),int(x1-30):int(x2+30)]
            
            in_img = trans(face)
            in_img = in_img.unsqueeze(0)
            in_img = in_img.to(device)
            
            out = model(in_img)
            prob = torch.exp(out)
            a =list(prob.squeeze())
            predicted = a.index(max(a))
            textSize = cv2.getTextSize(labels[predicted], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            textX = x1 + (x2-x1) // 2 - textSize[0] // 2
            cv2.putText(frame, labels[predicted], (int(textX), y1),cv2.FONT_HERSHEY_SIMPLEX, 0.7,labelColor[predicted],2)
    except (TypeError, ValueError) as e:
        pass
    
    cv2.imshow('Mask detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
    

