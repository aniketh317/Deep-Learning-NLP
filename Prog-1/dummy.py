#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:02:25 2022

@author: aniketh317
"""

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1,1)
    
    def forward(self, x):
        return self.l1(x)

model = Net()
y = 5
x = torch.tensor([2.000])
sav_dict = {}
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
optimizer.zero_grad()
for name, param in model.named_parameters():
    sav_dict[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
    print("Name:", name)
    print("Parameters:", param.data)
    print("Grad:", param.grad)

print('\n\n\n')
loss = (y-model(x))**2
loss.backward()
optimizer.step()

for name, param in model.named_parameters():
    print("Name:", name)
    print("New Parameters:", param.data)
    print("New Grad:", param.grad)
    

print('\n\n\n')
for k, v in sav_dict.items():
    print("Name:", k)
    print("Old Parameters", v[0].data)

with torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(sav_dict[name][0])
        param.grad = sav_dict[name][1]

print('\n\n\n')
for name, param in model.named_parameters():
    print("Name:", name)
    print("Parameters:", param)
    print("Grad:", param.grad)