import torch
from torch.utils.data import DataLoader
import pandas as pd
from main1 import CustomDataSet, collate_fn_padd
from model1 import Net


path = "Models/BERT_NO_TUNE"

testdata = CustomDataSet(pd.read_csv('Test Dataset.csv'))

model = Net()
fpath = path+"/model.pt"
model.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))

testloader = DataLoader(testdata, batch_size=200, num_workers=8, shuffle=False, collate_fn=collate_fn_padd)

model.eval()

acc = 0.0
for batch_idx, (Xb,lab,mask) in enumerate(testloader):
    pre = model.predict(src=Xb, src_mask=mask) #(N, )
    bacc = (torch.sum(pre==lab, dim=0))/(lab.shape[0])
    acc = acc + (bacc-acc)/(batch_idx+1)

print(acc)