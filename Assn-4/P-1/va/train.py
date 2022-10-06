import torch
import torch.nn as nn
from torch.utils.data import DataLoader

loss_function = nn.CrossEntropyLoss()  
    
def model_train(model,train_data, criterion, optimiser, collate_fn, embed=None, hidden=None, device = torch.device('cpu'), epoch_l=300, 
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
    vacc_time = torch.tensor([0.0 for i in range(0,11)]).to(device)
    vloader = DataLoader(vdata, batch_size=250, num_workers=8, shuffle=False, collate_fn=collate_fn)
    for batch_idx, (Xb,lab,lens) in enumerate(vloader):
        pre = model.predict(Xb.to(device), lens.to(device)) #(N,11)
        bacc_time = (torch.sum(pre==lab.to(device), dim=0))/(lab.shape[0])
        vacc_time = vacc_time + (bacc_time-vacc_time)/(batch_idx+1)
        bacc = torch.sum(bacc_time[0:10])/10
        vacc = vacc + (bacc-vacc)/(batch_idx+1)
    
    Macc = vacc
    Macc_time = vacc_time
    loss = 0.0

    tloader = DataLoader(tdata, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)
    for batch_idx, (Xb,lab,lens) in enumerate(tloader):
        b_loss = criterion(torch.transpose(model(Xb.to(device), lens.to(device)),1,2), lab.to(device))
        loss = loss+(b_loss-loss)/(batch_idx+1)
        
    epoch_vs_loss = []
    epoch_vs_loss.append(loss.item())
    save_weights = {}

    while epoch <= epoch_l and not stop:
        epoch += 1
        model.train()
        loss = 0.0

        tloader = DataLoader(tdata, batch_size=128, shuffle=True, num_workers=8, collate_fn=collate_fn)
        for batch_idx, (Xb,lab,lens) in enumerate(tloader):
            
            optimiser.zero_grad()
            b_loss = criterion(torch.transpose(model(Xb.to(device), lens.to(device)),1,2), lab.to(device))
            b_loss.backward()
            optimiser.step()

            loss = loss+(b_loss-loss)/(batch_idx+1)
        
        epoch_vs_loss.append(loss.item())

        model.eval()

        vacc = 0.0
        vacc_time = torch.tensor([0.0 for i in range(0,11)]).to(device)
        vloader = DataLoader(vdata, batch_size=250, num_workers=8, shuffle=False, collate_fn=collate_fn)
        for batch_idx , (Xb,lab,lens) in enumerate(vloader):
            pre = model.predict(Xb.to(device), lens.to(device)) #(N,11)
            bacc_time = (torch.sum(pre==lab.to(device), dim=0))/(lab.shape[0])
            vacc_time = vacc_time + (bacc_time-vacc_time)/(batch_idx+1)
            bacc = torch.sum(bacc_time[0:10])/10
            vacc = vacc + (bacc-vacc)/(batch_idx+1)
            
        if verbose:
            print('Embed:{}, Hidden:{} -- Epoch number:{}, training loss:{}, validation accuracy:{}'.format(embed, hidden, epoch+epoch_s, loss, vacc))
        
        if vacc>Macc:
            Macc = vacc
            Macc_time = vacc_time
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
    return Macc, Macc_time, epoch_vs_loss[0:num_epoch]