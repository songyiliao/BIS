import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import loaddata
from utils import PNet, Network, train
import torch.utils.data as Data

# 0: Company A; 1: Company B; 2: Company C; 3: Startup Company
t = 0

filename = ["df_A","df_B","df_C","df_sim"]
weight_filename = "weight"
load_path = 'BIS-main/'
file = load_path + filename + ".xlsx"
weightfile = load_path + weight_filename + ".xlsx"
save_path = 'saved_results/'
path_pnet = 'saved_models/%spnet.pt'% filename
path_net = 'saved_models/%snet.pt' % filename

# load data
data = pd.read_excel(file)
dataCols = data.columns
defVar = loaddata.defVar(dataCols)
X = data[ defVar.DummyVar() ]
Y = data[ defVar.YVar() ]
# load weight
weight_data = pd.read_excel(weightfile)
W = weight_data[ defVar.WghtVar() ]
if W.shape[0] == 1:
    W = pd.DataFrame(np.repeat(W.values, X.shape[0], axis = 0), columns = W.columns)

W.columns = X.columns


# input data into PNet
pnet = PNet(defVar.DummyVar(),defVar.WghtVar())
pnet_input = pnet.pd2tensor(X,W)
pnet_out = pnet.forwrd(pnet_input)

#the output of PNet is forward to the rest Network
net = Network(in_features = pnet_out.size(1), H1 = 30, H2 = 10)

# set hyper parameters
lr_list = [0.005,0.005,0.01,0.0005]
epochs_list = [200,200,200,200]
lr = lr_list[t]
epochs = epochs_list[t]
batch_size = 128
params = list(pnet.pnet_para)+list(net.parameters())
optimizer = torch.optim.Adam(params = params, lr = lr)
criterion = nn.MSELoss()

# generate training data
torch_dataset = Data.TensorDataset(pnet_input, torch.from_numpy(Y.values).double())

train_dataset, val_dataset, test_dataset = Data.random_split(torch_dataset, [0.7, 0.2, 0.1])
train_loader = Data.DataLoader(
    dataset = train_dataset, #torch TensorDataset format
    batch_size = batch_size)

valid_loader = Data.DataLoader(
    dataset = val_dataset, #torch TensorDataset format
    batch_size = batch_size)

test_loader = Data.DataLoader(
    dataset = test_dataset, #torch TensorDataset format
    batch_size = test_dataset.__len__())
for idx, (batch_x, batch_y) in enumerate(test_loader):
    testX = batch_x
    testy = batch_y
    torch.save(testX, save_path+'%stestX.pt'%filename)
    torch.save(testy, save_path+'%stesty.pt'%filename)
# train
min_valid_loss = np.inf

train_loss_list = torch.empty(epochs)
valid_loss_list = torch.empty(epochs)
test_loss_list = torch.empty(epochs)
loss = {"train_loss": train_loss_list,
        "valid_loss": valid_loss_list,
        "test_loss": test_loss_list}
for epoch in range(epochs):
    # epoch_loss = 0
    # correct = 0
    for idx, (batch_x, batch_y) in enumerate(train_loader):
        train_loss, train_predictions = train(pnet, net, batch_x, batch_y, optimizer, criterion)
    # print('Epoch {} Training Loss : {}'.format((epoch+1), train_loss))
    
    for idx, (batch_x, batch_y) in enumerate(valid_loader):
        _pnet_out = pnet.forwrd(batch_x)
        valid_predictions = net(_pnet_out)
        valid_loss = criterion(valid_predictions,batch_y)   
    # print('Epoch {} Validation Loss : {}'.format((epoch+1), valid_loss))
    
    # test
    _pnet_out = pnet.forwrd(testX)
    test_predictions = net(_pnet_out)
    test_loss = criterion(test_predictions,testy)

    if min_valid_loss > valid_loss:
        # print('Epoch {} Training Loss : {}'.format((epoch+1), train_loss))
        # print('Epoch {} Validation Loss : {}'.format((epoch+1), valid_loss))
        # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
        min_valid_loss = valid_loss
        # save the models
        torch.save(pnet, path_pnet)
        torch.save(net, path_net)
    
    train_loss_list[epoch] = train_loss
    valid_loss_list[epoch] = valid_loss
    test_loss_list[epoch] = test_loss

import pickle
with open(save_path+'%sloss.pkl'%filename, 'wb') as fp:
    pickle.dump(loss, fp)