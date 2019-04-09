#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:04:39 2019

@author: Kent Evans

        Testing using CIFAR10.  Input images are 32x32x3
        Conv layers:
            -Input
                number of channels(3 in RGB images)
                number of kernels to output (each kernel has identical depth to the input channel)
                spatial size of the kernels (can be tuple)
                stride, padding can be defined
            -Dimensions (ripped from cs231n)
                input is HxWxD, kernel is FxF, padding is P, stride is S
                output is ((H-F+2P)/S + 1)x((W-F+2P)/S + 1)xNumKernels (each kernel is FxFxD)

"""

import torch
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np

class VAE (nn.Module):
    def __init__(self):
        super().__init__()
        self.n_lat = 64
        #In: 32x32x3
        #Out: 32x32x16
        #Kernel: 3x3x3
        self.conv1 = nn.Conv2d( 3 , 16 , 3 , stride = 1 , padding = 1 ) 
        self.relu1 = nn.LeakyReLU()
        #In: 32x32x16
        #Out: 16x16x16
        #Kernel: 2x2, stride 2
        self.pool1 = nn.MaxPool2d( 2 , stride = 2 ) 
        #In: 16x16x16
        #Out: 16x16x32
        #Kernel: 3x3x16
        self.conv2 = nn.conv2d( 16 , 32 , 3 , stride = 1 , padding = 1 )
        self.relu2 = nn.LeakyReLU()
        #In: 16x16x32
        #Out: 8x8x32
        #Kernel: 2x2, stride 2
        self.pool2 = nn.MaxPool2d( 2 , stride = 2 )
        #Reshape here to be vector
        self.fcMean = nn.Linear( 32*8*8 , self.n_lat )
        self.fcVar = nn.Linear( 32*8*8 , self.n_lat )
        self.sample = nn.Linear( self.n_lat , 32*8*8 )
        
        
    def forward( self , img ):
        img = self.pool1(self.relu1(self.conv1(img)))
        img = self.pool2(self.relu2(self.conv2(img)))
        mean = self.fcMean(img.view(-1,32*8*8))
        var = self.fcVar(img.view(-1,32*8*8))
        return mean , var
    
    def sample( self , mean , var ):
        std = torch.exp(.5, var)
        ran = torch.rand(std.size(0),-1)
        return mean + (std*ran)
    
    