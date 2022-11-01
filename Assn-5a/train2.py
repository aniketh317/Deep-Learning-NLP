from numpy import dtype
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

loss_function = nn.CrossEntropyLoss()  
    
def model_train(model,train_data, criterion, optimiser, collate_fn, device = torch.device('cpu'), epoch_l=300, 
                epoch_s=0, verbose=False, patience=40):

    #85-15 split train,valid
    tlen = int(0.85*len(train_data))
    vlen = len(train_data)-tlen
    lengths = [tlen, vlen]
    tdata, vdata = torch.utils.data.random_split(train_data, lengths)

    obs = 0
    stop = False
    epoch = 0
    num_epoch = 0

    model.eval()
    
    vacc = 0.0
    vloader = DataLoader(vdata, batch_size=250, num_workers=0, shuffle=False, collate_fn=collate_fn)
    for batch_idx, (Xb,lab,mask) in enumerate(vloader):
        pre = model.predict(src=Xb.to(device), src_mask=mask.to(device)) #(N, )
        bacc = (torch.sum(pre==lab.to(device), dim=0))/(lab.shape[0])
        vacc = vacc + (bacc-vacc)/(batch_idx+1)
    
    Macc = vacc
    loss = 0.0

    tloader = DataLoader(tdata, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn)
    for batch_idx, (Xb,lab,mask) in enumerate(tloader):
        b_loss = criterion(model(src=Xb.to(device), src_mask=mask.to(device)), lab.to(device))
        loss = loss+(float(b_loss)-loss)/(batch_idx+1)
        
    epoch_vs_loss = []
    epoch_vs_loss.append(loss)
    save_weights = {}

    while epoch <= epoch_l and not stop:
        epoch += 1
        model.train()
        loss = 0.0

        tloader = DataLoader(tdata, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn)
        for batch_idx, (Xb,lab,mask) in enumerate(tloader):
            
            optimiser[0].zero_grad()
            optimiser[1].zero_grad()
            b_loss = criterion(model(src=Xb.to(device), src_mask=mask.to(device)), lab.to(device))
            b_loss.backward()
            optimiser[0].step()
            optimiser[1].step()

            loss = loss+(float(b_loss)-loss)/(batch_idx+1)
        
        epoch_vs_loss.append(loss)

        model.eval()

        vacc = 0.0
        vloader = DataLoader(vdata, batch_size=250, num_workers=8, shuffle=False, collate_fn=collate_fn)
        for batch_idx , (Xb,lab,mask) in enumerate(vloader):
            pre = model.predict(src=Xb.to(device), src_mask=mask.to(device)) #(N,)
            bacc = (torch.sum(pre==lab.to(device), dim=0))/(lab.shape[0])
            vacc = vacc + (bacc-vacc)/(batch_idx+1)
            
        if verbose:
            print('Epoch number:{}, training loss:{}, validation accuracy:{}'.format(epoch+epoch_s, loss, vacc))
        
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