#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:27:04 2022

@author: aniketh317
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,dim, prob):
        super(Net, self).__init__()
        self.dim = dim
        
        self.g = nn.parameter.Parameter(torch.rand(1)) #Glove embedding multiplier
        self.f = nn.parameter.Parameter(torch.rand(1)) #FastText multiplier

        self.l1 = nn.Linear(dim, 2*dim)
        self.d1 = nn.Dropout(p=prob)
        self.l2 = nn.Linear(2*dim, 2*dim)
        self.d2 = nn.Dropout(p=prob)
        self.l3 = nn.Linear(2*dim, 2*dim)
        self.d3 = nn.Dropout(p=prob)
        self.l4 = nn.Linear(2*dim,2)
    
    def forward(self, x):
        out = (self.g)*x[:,0:self.dim] + (self.f)*x[:,self.dim:2*self.dim]
        out = self.l1(out)
        out = F.relu(out)
        out = self.d1(out)
        out = self.l2(out)
        out = F.relu(out)
        out = self.d2(out)
        out = self.l3(out)
        out = F.relu(out)
        out = self.d3(out)
        out = self.l4(out)
        return out 
    
    def predict(self, x):
        out = self.forward(x)
        prediction = torch.argmax(out,dim=1)
        return prediction

    
