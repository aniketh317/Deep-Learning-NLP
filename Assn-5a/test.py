import torch
from torch.utils.data import DataLoader
import pandas as pd
from main import CustomDataSet, collate_fn_padd
from model import Net

edim = 300
num_head = 5

path = "Models/Embed-{}_Heads-{}".format(edim,num_head)

testdata = CustomDataSet(pd.read_csv('Test Dataset.csv'), edim=edim)

model = Net(vocab=testdata.embeddings.shape[0], edim=edim, n_heads=num_head)
fpath = path+"/model.pt"
model.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))

testloader = DataLoader(testdata, batch_size=200, shuffle=False, collate_fn=collate_fn_padd)

model.eval()

acc = 0.0
for batch_idx, (Xb,lab,mask) in enumerate(testloader):
    pre = model.predict(src=Xb, src_mask=mask) #(N, )
    bacc = (torch.sum(pre==lab, dim=0))/(lab.shape[0])
    acc = acc + (bacc-acc)/(batch_idx+1)

print(acc)