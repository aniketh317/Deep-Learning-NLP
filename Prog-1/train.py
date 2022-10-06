#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:43:28 2022

@author: aniketh317
"""

import torch
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #use cuda:0

def train(model, loss_criterion, train_loader, validation_loader=None, optimizer=None, epochs_range=100, min_delta=0, patience=10,
          verbose=False):
    """train_loader -> Data Loader for the trained data, validation_loader -> Data loader for the validate data"""

    if not optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    if not validation_loader:
        validation_loader = train_loader #ToDo
    
    valid_iter = iter(validation_loader)
    
    parameter_dict = {}
    model.eval() #Evaluation mode
    (valid_X, valid_y) = next(valid_iter)
    valid_out = model(valid_X.to(device))
    #import pdb; pdb.set_trace()
    valid_loss = loss_criterion(valid_out, valid_y.to(device))
    min_loss = valid_loss
    save_weights = {}
    
    for name, param in model.named_parameters():
        save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
    
    epochs=0
    stop=False
    obs = 0 #Number of not sufficient validation loss decreases observed, so far
    num_epochs = 0 #Number of epochs
    while epochs < epochs_range and not stop:
        train_loss = torch.tensor(0.0)
        model.train() #Training mode
        batches = 0
        for batch_idx, (train_X, train_y) in enumerate(train_loader):
            batches += 1
            #zero the accumulated gradients in grad attributes of parameters
            optimizer.zero_grad()
            
            train_out = model(train_X.to(device))
            loss = loss_criterion(train_out, train_y.to(device))
            loss.backward()
            train_loss = train_loss+(loss-train_loss)*(1/batches)
            optimizer.step()
        epochs += 1
        
        model.eval()
        
        valid_iter = iter(validation_loader)
        (valid_X, valid_y) = next(valid_iter)
        valid_out = model(valid_X.to(device))
        valid_loss = loss_criterion(valid_out.to(device), valid_y.to(device))
        
        if(verbose):
            print('Epochs - {}, Validation Loss - {}, Train Loss - {}'.format(epochs, valid_loss, train_loss))
        
        if valid_loss+min_delta >= min_loss:
            obs += 1
            if obs > patience:
                stop = True            
        else:
            obs = 0
            min_loss = valid_loss
            for name, param in model.named_parameters():
                save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
            num_epochs = epochs
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(save_weights[name][0])
            param.grad = save_weights[name][1]

    return num_epochs, valid_loss
                
        
            
            
        
            