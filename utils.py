import torch.nn as nn
import torch
import pandas as pd

class PNet:
    def __init__(self,list_DummyVar,list_WghtVar):
        if len(list_DummyVar) != len(list_WghtVar):
            "features and weights have different dimension"
    def pd2tensor(self,X,W): # X is the enabler and W is the weights 
        categories = X.values.max()+1
        input_size = (X.shape[0],X.shape[1]*categories)
        input = torch.empty(size = input_size)
        col_idx = [0]
        for idx,_x in X.items():
            x = pd.get_dummies(_x)
            w = W[idx]
            x_tensor = torch.from_numpy(x.values).double()
            w_tensor = torch.from_numpy(w.values).double()

            input[:,col_idx[-1]] = w_tensor
            input[:,col_idx[-1]+1:col_idx[-1]+x_tensor.size(1)+1] = x_tensor

            newcol = col_idx[-1]+x_tensor.size(1)+1
            col_idx.append(newcol) # save column No. for each enabler
        input = input[:,0:col_idx[-1]]  # delete extra columns
        self.pnet_para = torch.rand(size = (col_idx[-1],1))
        self.col_idx = col_idx
        return input
       
    def forwrd(self, input):
        col_idx = self.col_idx
        pnet_para = self.pnet_para
        pnet_out = torch.empty(size = (input.size(0),len(col_idx)-1))
        for t in range(len(col_idx)-1):
            _x = input[:,col_idx[t]:col_idx[t+1]]
            _w = pnet_para[col_idx[t]:col_idx[t+1],:]
            pnet_out[:,t:t+1] = torch.mm(_x,_w)
        return pnet_out

class Network(nn.Module):
    def __init__(self, in_features, out_features = 1, H1 = 30, H2 = 10):
        super().__init__()
        layer1 = nn.Sequential()
        layer1.add_module('fc1',nn.Linear(in_features, H1))
        layer1.add_module('relu', nn.ReLU())
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('fc2',nn.Linear(H1, H2))
        layer2.add_module('drop', nn.Dropout(p=0.2))      
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('fc3',nn.Linear(H2, out_features))
        self.layer3 = layer3

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def train(PNet, Network, pnet_input, y, optimizer, criterion):
    optimizer.zero_grad()

    pnet_out = PNet.forwrd(pnet_input)   
    output = Network(pnet_out.float())

    loss = criterion(output,y.float())

    loss.backward()
    optimizer.step()

    return loss, output

            
    
