# PyTorch
from turtle import forward
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
#from selfonn import SelfONNLayer
import timm

def reset_function_generic(m):
    if hasattr(m,'reset_parameters') or hasattr(m,'reset_parameters_like_torch'): 
        # print(m) 
        if isinstance(m, SelfONNLayer):
            m.reset_parameters_like_torch() 
        else:
            m.reset_parameters()

class SqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x 

class UnSqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.unsqueeze(2).unsqueeze(3)
        return x 



class CNN_Classifier(nn.Module):
    def __init__(self,in_channels,class_num):
        super().__init__()
        
        # # example 1
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_channels, 256), 
        #     nn.Dropout(0.2), 
        #     nn.ReLU(), 
        #     nn.Linear(256, class_num), 
        #     nn.LogSoftmax(dim=1) 
        #     )
        # torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        # self.classifier[0].bias.data.fill_(0.01) 
        # torch.nn.init.xavier_uniform_(self.classifier[2].weight)
        # self.classifier[2].bias.data.fill_(0.01) 

        # example 2 
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, class_num), 
            nn.LogSoftmax(dim=1) 
            )
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.01) 

        # example 3
        # self.classifier = nn.Sequential(
        #     UnSqueezeLayer(),
        #     nn.Tanh(),
        #     SelfONNLayer(in_channels=in_channels,out_channels=class_num,kernel_size=1,q=3,mode='fast',dropout=None),
        #     SqueezeLayer(),
        #     nn.LogSoftmax(dim=1) 
        #     ) 

        # # example 4
        # self.classifier = nn.Sequential(
        #     UnSqueezeLayer(),
        #     nn.Tanh(),
        #     SelfONNLayer(in_channels=in_channels,out_channels=256,kernel_size=1,q=1,mode='fast',dropout=0.2),
        #     nn.Tanh(), 
        #     SelfONNLayer(in_channels=256,out_channels=class_num,kernel_size=1,q=1,mode='fast',dropout=None),
        #     SqueezeLayer(),
        #     nn.LogSoftmax(dim=1) 
        #     )
     

    def forward(self,x):
        x = self.classifier(x)
        return x




class Self_B_ResBlock(nn.Module):
    def __init__(self, in_channels=3, channel1=8, channel2=16, channel3=8, resConnection=False,q_order=3):
        super().__init__()
        self.layer1 = SelfONNLayer(in_channels=in_channels,out_channels=channel1,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch1 = nn.BatchNorm2d(channel1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.resConnection = resConnection
        if self.resConnection:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        else:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = SelfONNLayer(in_channels=channel2,out_channels=channel3,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch3 = nn.BatchNorm2d(channel3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()

    def forward(self,x):
        input = x.clone()
        x = self.tanh(self.Batch1(self.layer1(x)))
        x = self.tanh(self.Batch2(self.layer2(x)))
        x = self.tanh(self.Batch3(self.layer3(x)))
        if self.resConnection:
            x= self.tanh(x.clone()+ input)
            
            # x = torch.cat((x,input), 1)
        return x

class Self_MobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10):
        super().__init__()
        self.class_num = class_num
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=16, channel2=48, channel3=32, resConnection=True)
        self.NonResidual1 = Self_B_ResBlock(in_channels=32, channel1=32, channel2=48, channel3=16, resConnection=False)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=48, channel3=16, resConnection=True)
        self.NonResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=32, channel3=48, resConnection=False)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=48, channel1=16, channel2=56, channel3=48, resConnection=True)
        self.NonResidual3 = Self_B_ResBlock(in_channels=48, channel1=8, channel2=48, channel3=36, resConnection=False)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=36, channel1=16, channel2=32, channel3=36, resConnection=True)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=36, resConnection=True)
        self.NonResidual4 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=last_layer_channel, resConnection=False)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7,7))
        self.Flatten = nn.Flatten()
        self.Dropout2d = nn.Dropout(p=0.1)
        self.self_MLP = CNN_Classifier(in_channels = int(49*last_layer_channel),class_num = self.class_num)
        
    def forward(self, x):
        x= self.tanh(self.batchnorm(self.selfONN(x)))
        x= self.BottleneckResidual1(x)
        x= self.NonResidual1(x)
        x= self.BottleneckResidual2(x)
        x= self.NonResidual2(x) 
        x= self.BottleneckResidual3(x)
        x= self.NonResidual3(x)
        x= self.BottleneckResidual4(x)
        x= self.BottleneckResidual5(x)
        x= self.NonResidual4(x)
        x= self.AdaptiveAvgPool2d(x)
        x = self.Flatten(x)
        x = self.Dropout2d(x)
        x= self.self_MLP(x)
        return x



class Self_DenseMobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10, q_order=3):
        super().__init__()
        self.class_num = class_num
        self.InputMLP = 10
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.maxpool = nn.MaxPool2d(2)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual6 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual7 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual8 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(p=0.2)
        self.self_MLP = CNN_Classifier(in_channels =5*32,class_num = self.class_num)
    
    def forward(self,x):
        x = self.tanh(self.batchnorm(self.selfONN(x)))
        x = self.BottleneckResidual1(x)
        x = self.BottleneckResidual2(x)
        ##############
        inMLP1 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual3(self.maxpool(x))
        x = self.BottleneckResidual4(x)
        ##############
        inMLP2 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual5(self.maxpool(x))
        x = self.BottleneckResidual6(x)
        ##############
        inMLP3 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP4 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP5 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        inMLP = torch.cat((inMLP1, inMLP2, inMLP3, inMLP4, inMLP5), dim=1)
        
        
        x= self.Flatten(inMLP)
        
        x = self.Dropout(x)
        x= self.self_MLP(x)
        return x




def cnn_V1(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 10th layer (MLP)
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 11th layer (MLP)  
        torch.nn.Linear(in_features=4096, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )  
    #
    return model 

def cnn_V2(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=256, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.2, inplace=False),
        # 10th layer (MLP)   
        torch.nn.Linear(in_features=256, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )
    #
    return model 

def cnn_V3(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    )
    #
    return model 


def cnn_V4(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 20, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 2nd layer (MLP)
        # conv_output = 7*7*20= 980 
        torch.nn.Linear(in_features=980, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    )
    #
    return model 


class cnn_V5(nn.Module):
    
    def __init__(self, input_ch, class_num): 
        super(cnn_V5, self).__init__() 

        # 1st layer (conv)
        self.conv1 = cnn_V5.conv_block(in_channels=input_ch, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Average pooling 
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = torch.nn.Flatten()
        # 2nd layer (MLP) 
        # conv_output = 7*7*20= 980
        self.MLP2 = torch.nn.Linear(in_features=980, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.pool1(layer1)
        Pool_layer = self.AvgPool(layer1)
        Pool_layer = self.flatten(Pool_layer)
        Output_layer = self.MLP2(Pool_layer) 
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):     
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            torch.nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True) 
        )

def SelfONN_1(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   

        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=75,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(75, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=75,out_channels=56,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(),  
    
        # Output layer (MLP)  
        
        torch.nn.Linear(in_features=2744, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

# def SelfONN_1(input_ch, class_num, q_order): 
#     model = torch.nn.Sequential(   
#         # 1st layer (conv) 
#         SelfONNLayer(in_channels=input_ch,out_channels=2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(4),
#         torch.nn.Tanh(),
#         # 2nd layer (conv)
#         SelfONNLayer(in_channels=2,out_channels=4,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(4),  
#         torch.nn.Tanh(), 
#         # flatten 
#         torch.nn.Flatten(),  
#         # Output layer (MLP)  
#         torch.nn.Linear(in_features=576, out_features=class_num, bias=True),  
#         torch.nn.LogSoftmax(dim=1)
#     ) 
#     #
#     reset_fn = reset_function_generic 
#     model.apply(reset_fn) 
#     return model 


def SelfONN_2(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=784, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_2_dense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(),  
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 



def SelfONN_3(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        #torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)
        torch.nn.Dropout(p=0.2, inplace=False),  
        torch.nn.Linear(in_features=1568, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

def SelfONN_3_SelfDense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 



def SelfONN_4(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_5(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 7th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 8th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

def get_pretrained_model(parentdir, model_name,ImageNet,input_ch,class_num,train_on_gpu,multi_gpu,q_order):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """
  

    if model_name == 'cnn_V1': 
        model = cnn_V1(input_ch,class_num)  
    
    elif model_name == 'cnn_V2': 
        model = cnn_V2(input_ch,class_num)  

    elif model_name == 'cnn_V3': 
        model = cnn_V3(input_ch,class_num)  

    elif model_name == 'cnn_V4': 
        model = cnn_V4(input_ch,class_num)  

    elif model_name == 'cnn_V5':   
        model = cnn_V5(input_ch,class_num)    

    elif model_name == 'SelfONN_1': 
        model = SelfONN_1(input_ch, class_num, q_order) 
    
    elif model_name == 'SelfONN_2': 
        model = SelfONN_2(input_ch, class_num, q_order) 

    elif model_name == 'SelfONN_3':  
        model = SelfONN_3(input_ch, class_num, q_order) 

    elif model_name == 'SelfONN_3_SelfDense':  
        model = SelfONN_3_SelfDense(input_ch, class_num, q_order) 

    elif model_name == 'SelfONN_2_dense':  
        model = SelfONN_2_dense(input_ch, class_num, q_order) 

    elif model_name == 'Self_MobileNet':
        model = Self_MobileNet(input_channel = input_ch, last_layer_channel = 32, class_num= class_num)

    elif model_name == 'Self_DenseMobileNet':
        model =Self_DenseMobileNet(input_channel = input_ch, last_layer_channel = 32, class_num= class_num, q_order=q_order)

    elif model_name == 'squeezenet1_0':
        from squeezenet import squeezenet1_0
        model = squeezenet1_0(pretrained=ImageNet) 
        model.classifier[-1] = nn.Sequential(  
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),  
            CNN_Classifier(in_channels=1000,class_num=class_num)
            )   
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=ImageNet) 
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=ImageNet)
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=ImageNet)
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=ImageNet)
        n_inputs = model.fc.in_features 
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'inception_v3':
        from Inception_Networks import inception_v3
        model = inception_v3(pretrained=ImageNet) 
        num_ftrs_Aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_Aux, class_num) 
        # Handle the primary net
        n_inputs = model.fc.in_features 
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)


    elif model_name == 'inception_v4':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context   
        n_inputs = model.last_linear.in_features 
        model.last_linear = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'inceptionresnetv2':
        from inceptionresnetv2 import inceptionresnetv2
        model = inceptionresnetv2(parentdir, num_classes=1000, pretrained='imagenet')
        n_inputs = model.last_linear.in_features
        model.last_linear = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'xception':
        from xception import xception
        model = xception(pretrained=ImageNet)
        n_inputs = model.fc.in_features
        model.fc =CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    elif model_name == 'chexnet':
        from chexnet import chexnet
        model = chexnet(parentdir) 
        n_inputs = model.module.densenet121.classifier[0].in_features
        model.module.densenet121.classifier = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'nasnetalarge':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        n_inputs = model.last_linear.in_features
        model.last_linear = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    if model_name == 'pnasnet5large':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        n_inputs = model.last_linear.in_features
        model.last_linear = CNN_Classifier(in_channels=n_inputs,class_num=class_num) 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    if model_name == 'densenet121':
        model = models.densenet121(pretrained=ImageNet) 
        n_inputs = model.classifier.in_features
        model.classifier = CNN_Classifier(in_channels=n_inputs,class_num=class_num)  

    if model_name == 'densenet161':
        model = models.densenet161(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.classifier.in_features
        model.classifier = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'densenet201': 
        model = models.densenet201(pretrained=ImageNet)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier.in_features
        model.classifier = CNN_Classifier(in_channels=n_inputs,class_num=class_num) 

    if model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'googlenet':
        model = models.googlenet(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'nasnetamobile':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = CNN_Classifier(in_channels=n_inputs,class_num=class_num)

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=ImageNet)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = CNN_Classifier(in_channels=n_inputs,class_num=class_num) 

    if model_name == 'darknet53':
        from darknet53 import darknet53 
        model = darknet53(1000)
        checkpoint = parentdir + 'models/darknet53.pth.tar'
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint ['state_dict'])
        del checkpoint 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.fc.in_features 
        model.fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 


    elif model_name == 'efficientnet_b0':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b1':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b1') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b2':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b2') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b3':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b4':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b5':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b5') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b6':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b6') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'efficientnet_b7':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7') 
        n_inputs = model._fc.in_features 
        model._fc = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        model._swish =  nn.Identity() 
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 

    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        n_inputs = model.classifier[3].in_features
        model.classifier[3] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 
    
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=True)
        n_inputs = model.classifier[3].in_features
        model.classifier[3] = CNN_Classifier(in_channels=n_inputs,class_num=class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilenetv3_rw':
        model = timm.create_model("mobilenetv3_rw", pretrained=ImageNet)
        n_inputs = model.classifier.in_features
        model.classifier = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn) 
    
    
    elif model_name == 'convit_base':
        model = timm.create_model("convit_base", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_xlarge_384_in22ft1k':
        model = timm.create_model("convnext_xlarge_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    
    elif model_name == 'convit_small':
        model = timm.create_model("convit_small", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convit_tiny':
        model = timm.create_model("convit_tiny", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convmixer_768_32':
        model = timm.create_model("convmixer_768_32", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convmixer_1024_20_ks9_p14':
        model = timm.create_model("convmixer_1024_20_ks9_p14", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convmixer_1536_20':
        model = timm.create_model("convmixer_1536_20", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_atto':
        model = timm.create_model("convnext_atto", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_atto_ols':
        model = timm.create_model("convnext_atto_ols", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_base':
        model = timm.create_model("convnext_base", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_base_384_in22ft1k':
        model = timm.create_model("convnext_base_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_base_in22ft1k':
        model = timm.create_model("convnext_base_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_base_in22k':
        model = timm.create_model("convnext_base_in22k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_femto':
        model = timm.create_model("convnext_femto", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_femto_ols':
        model = timm.create_model("convnext_femto_ols", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_large':
        model = timm.create_model("convnext_large", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_large_384_in22ft1k':
        model = timm.create_model("convnext_large_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_large_in22ft1k':
        model = timm.create_model("convnext_large_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_large_in22k':
        model = timm.create_model("convnext_large_in22k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_nano':
        model = timm.create_model("convnext_nano", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_nano_ols':
        model = timm.create_model("convnext_nano_ols", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_pico':
        model = timm.create_model("convnext_pico", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_pico_ols':
        model = timm.create_model("convnext_pico_ols", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_small':
        model = timm.create_model("convnext_small", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_small_384_in22ft1k':
        model = timm.create_model("convnext_small_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_small_in22ft1k':
        model = timm.create_model("convnext_small_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_small_in22k':
        model = timm.create_model("convnext_small_in22k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_tiny':
        model = timm.create_model("convnext_tiny", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_tiny_384_in22ft1k':
        model = timm.create_model("convnext_tiny_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_tiny_hnf':
        model = timm.create_model("convnext_tiny_hnf", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_tiny_in22ft1k':
        model = timm.create_model("convnext_tiny_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_tiny_in22k':
        model = timm.create_model("convnext_tiny_in22k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_xlarge_384_in22ft1k':
        model = timm.create_model("convnext_xlarge_384_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_xlarge_in22ft1k':
        model = timm.create_model("convnext_xlarge_in22ft1k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'convnext_xlarge_in22k':
        model = timm.create_model("convnext_xlarge_in22k", pretrained=ImageNet)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)

    
    elif model_name == 'mobilevit_s':
        model = timm.create_model("mobilevit_s", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevit_xs':
        model = timm.create_model("mobilevit_xs", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevit_xxs':
        model = timm.create_model("mobilevit_xxs", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_050':
        model = timm.create_model("mobilevitv2_050", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_075':
        model = timm.create_model("mobilevitv2_075", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_100':
        model = timm.create_model("mobilevitv2_100", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_125':
        model = timm.create_model("mobilevitv2_125", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_150_384_in22ft1k':
        model = timm.create_model("mobilevitv2_150_384_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_150_in22ft1k':
        model = timm.create_model("mobilevitv2_150_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_175':
        model = timm.create_model("mobilevitv2_175", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_175_384_in22ft1k':
        model = timm.create_model("mobilevitv2_175_384_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_175_in22ft1k':
        model = timm.create_model("mobilevitv2_175_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_200':
        model = timm.create_model("mobilevitv2_200", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_200_384_in22ft1k':
        model = timm.create_model("mobilevitv2_200_384_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mobilevitv2_200_in22ft1k':
        model = timm.create_model("mobilevitv2_200_in22ft1k", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mvitv2_base':
        model = timm.create_model("mvitv2_base", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mvitv2_large':
        model = timm.create_model("mvitv2_large", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mvitv2_small':
        model = timm.create_model("mvitv2_small", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'mvitv2_tiny':
        model = timm.create_model("mvitv2_tiny", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)




    elif model_name == 'visformer_small':
        model = timm.create_model("visformer_small", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch8_224':
        model = timm.create_model("vit_base_patch8_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch8_224_dino':
        model = timm.create_model("vit_base_patch8_224_dino", pretrained=True)
        n_inputs = 768
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch8_224_in21k':
        model = timm.create_model("vit_base_patch8_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    


    elif model_name == 'vit_base_patch16_224':
        model = timm.create_model("vit_base_patch16_224", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        # if not ImageNet:
        #     reset_fn = reset_function_generic 
        #     model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_224_dino':
        model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
        n_inputs = 768
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_224_in21k':
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_224_miil':
        model = timm.create_model("vit_base_patch16_224_miil", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_224_miil_in21k':
        model = timm.create_model("vit_base_patch16_224_miil_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_224_sam':
        model = timm.create_model("vit_base_patch32_224_sam", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_384':
        model = timm.create_model("vit_base_patch16_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch16_rpn_224':
        model = timm.create_model("vit_base_patch16_rpn_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_224':
        model = timm.create_model("vit_base_patch32_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_224_clip_laion2b':
        model = timm.create_model("vit_base_patch32_224_clip_laion2b", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_224_in21k':
        model = timm.create_model("vit_base_patch32_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_224_sam':
        model = timm.create_model("vit_base_patch32_224_sam", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_patch32_384':
        model = timm.create_model("vit_base_patch32_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_r50_s16_224_in21k':
        model = timm.create_model("vit_base_r50_s16_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_base_r50_s16_384':
        model = timm.create_model("vit_base_r50_s16_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_base_patch16_224':
        model = timm.create_model("vit_relpos_base_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_base_patch16_clsgap_224':
        model = timm.create_model("vit_relpos_base_patch16_clsgap_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_base_patch32_plus_rpn_256':
        model = timm.create_model("vit_relpos_base_patch32_plus_rpn_256", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_medium_patch16_224':
        model = timm.create_model("vit_relpos_medium_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_medium_patch16_cls_224':
        model = timm.create_model("vit_relpos_medium_patch16_cls_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_medium_patch16_rpn_224':
        model = timm.create_model("vit_relpos_medium_patch16_rpn_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_relpos_small_patch16_224':
        model = timm.create_model("vit_relpos_small_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_patch16_224_in21k':
        model = timm.create_model("vit_small_patch16_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_patch16_384':
        model = timm.create_model("vit_small_patch16_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_patch32_224':
        model = timm.create_model("vit_small_patch32_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_patch32_224_in21k':
        model = timm.create_model("vit_small_patch32_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_patch32_384':
        model = timm.create_model("vit_small_patch32_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_r26_s32_224':
        model = timm.create_model("vit_small_r26_s32_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_r26_s32_224_in21k':
        model = timm.create_model("vit_small_r26_s32_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_small_r26_s32_384':
        model = timm.create_model("vit_small_r26_s32_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_srelpos_medium_patch16_224':
        model = timm.create_model("vit_srelpos_medium_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_srelpos_small_patch16_224':
        model = timm.create_model("vit_srelpos_small_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_patch16_224':
        model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_patch16_224_in21k':
        model = timm.create_model("vit_tiny_patch16_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_patch16_384':
        model = timm.create_model("vit_tiny_patch16_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_r_s16_p8_224':
        model = timm.create_model("vit_tiny_r_s16_p8_224", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_r_s16_p8_224_in21k':
        model = timm.create_model("vit_tiny_r_s16_p8_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    elif model_name == 'vit_tiny_r_s16_p8_384':
        model = timm.create_model("vit_tiny_r_s16_p8_384", pretrained=True)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)






    elif model_name == 'vit_huge_patch14_224_in21k':
        model = timm.create_model("vit_huge_patch14_224_in21k", pretrained=True)
        n_inputs = model.head.in_features
        model.head = torch.nn.Sequential(
          torch.nn.Flatten(),
          CNN_Classifier(n_inputs, class_num),
          )

        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    
    

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)

    return model 







# class CustomLayer(nn.Module):
#             def __init__(self,layer_idx=None,in_channels=1,out_channels=1,kernel_size=1,sampling_factor=1,optimize=True):
#                 super().__init__() 
#                 self.in_channels = in_channels 
#                 self.out_channels = out_channels
#                 self.kernel_size = kernel_size
#                 self.sampling_factor = sampling_factor
#                 self.layer_idx = layer_idx
#             def forward(self,x): 
#                 x = x.squeeze(3) 
#                 x = x.squeeze(2)  
#                 return x 
