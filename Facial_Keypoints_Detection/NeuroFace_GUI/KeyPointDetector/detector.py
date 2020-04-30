import torch 
import torch.nn as nn
import torch.functional as F
import cv2
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim 
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import torchvision.utils as imutils

import pandas as pd
import numpy as np
import os
from PIL import Image

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchsummary import summary

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(devices))

print("PyTorch Version: {}".format(torch.__version__))
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()

    self.features = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5,5), stride=1, padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2,2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2,2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2,2),
        
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(512)
    )

    self.regressor = nn.Sequential(
        nn.Linear(512*6*6,256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.4),
        nn.Linear(256,30)
    ) 
    
  def forward(self, input_):
    input_ = self.features(input_)
    input_ = input_.view(input_.size(0),-1)
    return self.regressor(input_)


transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5],std=[0.5])
])
ind_even = [i for i in range(0,30,2)]
ind_odd = [i for i in range(1,30,2)]

class Detector:
    def __init__(self,path,device=devices,dimension=(96,96)):
        self.model = Net()
        self.device = device
        self.dimension = dimension
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transforms = transform
        try:
            self.model.load_state_dict(torch.load(path))
            print("Weights Loaded Successfully")
        except:
            print("Weights File couldn't be found")
    
    def detect(self,image):
        with torch.no_grad():
            image = self.transform(image)
            image = image.unsqueeze(0)
            image = image.to(self.device)
            prediction = self.model(image)
            prediction = prediction.squeeze()
            prediction = prediction.cpu().numpy()/self.dimension[0]
            X = prediction[ind_even]
            Y = prediction[ind_odd]
            return X,Y

    def transform(self,img):
        img = cv2.equalizeHist(img)
        img = np.array(img).astype(float)
        img = cv2.resize(img, self.dimension, interpolation = cv2.INTER_AREA)
        img = Image.fromarray(img)

        if self.transforms:
            img = self.transforms(img)
        return img
