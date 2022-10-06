import torch
import torch.nn as nn
import pickle
from model import Net
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

loss_function = nn.CrossEntropyLoss()  

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #if gpu presesnt cuda:0 else cpu only
torch.backends.cudnn.benchmark = True

    
def model_train(model,train_data, criterion, optimiser, collate_fn, epoch_l=300, verbose=False, patience=40):

    #80-20 split train,valid
    tlen = int(0.8*len(train_data))
    vlen = len(train_data)-tlen
    lengths = [tlen, vlen]
    tdata, vdata = torch.utils.data.random_split(train_data, lengths)

    obs = 0
    stop = False
    epoch = 0
    num_epoch = 0

    model.eval()
    
    vacc = 0.0
    vloader = DataLoader(vdata, batch_size=500, num_workers=8, shuffle=False, collate_fn=collate_fn)
    for batch_idx, (Xb, lab) in enumerate(vloader):
        pre = model.predict(Xb.to(device))
        bacc = (torch.sum(pre==lab.to(device)))/lab.shape[0]
        vacc = vacc + (bacc-vacc)/(batch_idx+1)
    
    Macc = vacc

    loss = 0.0

    tloader = DataLoader(tdata, batch_size=256, shuffle=True, num_workers=8, collate_fn=collate_fn)
    for batch_idx, (Xb,lab) in enumerate(tloader):
        b_loss = criterion(model(Xb.to(device)), lab.to(device))
        loss = loss+(b_loss-loss)/(batch_idx+1)
        
    epoch_vs_loss = []
    epoch_vs_loss.append(loss.item())
    save_weights = {}

    while epoch <= epoch_l and not stop:
        epoch += 1
        model.train()
        loss = 0.0

        tloader = DataLoader(tdata, batch_size=256, shuffle=True, num_workers=8, collate_fn=collate_fn)
        for batch_idx, (Xb,lab) in enumerate(tloader):
            
            optimiser.zero_grad()
            b_loss = criterion(model(Xb.to(device)), lab.to(device))
            b_loss.backward()
            optimiser.step()

            loss = loss+(b_loss-loss)/(batch_idx+1)
        
        epoch_vs_loss.append(loss.item())

        model.eval()

        vacc = 0.0
        vloader = DataLoader(vdata, batch_size=500, num_workers=8, shuffle=False, collate_fn=collate_fn)
        for batch_idx , (Xb, lab) in enumerate(vloader):
            pre = model.predict(Xb.to(device))
            bacc = (torch.sum(pre==lab.to(device)))/lab.shape[0]
            vacc = vacc + (bacc-vacc)/(batch_idx+1)
            
        if verbose:
            print('Epoch number:{}, training loss:{}, validation accuracy:{}'.format(epoch, loss, vacc))
        
        if vacc>Macc:
            Macc = vacc
            for name, param in model.named_parameters():
                save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
            num_epoch = epoch
            obs = 0

        else:
            obs += 1
            if obs > patience:
                stop = True
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(save_weights[name][0])
            param.grad = save_weights[name][1]
    
    print('Number of epochs considered:{}'.format(num_epoch))
    return Macc, epoch_vs_loss[0:num_epoch]