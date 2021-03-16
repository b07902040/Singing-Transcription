import json
import os
import numpy as np
import pickle
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

class Myrnn(nn.Module):
    def __init__(self, input_dim, hidden_size= 100):
        super(Myrnn, self).__init__()
        self.hidden_size = hidden_size

        self.Linear1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers= 5, bidirectional= True)
        self.Linear2 = nn.Linear(hidden_size* 2, 2)
        self.Linear3 = nn.Linear(hidden_size* 2, 1)

    def forward(self, input_data):
        out = F.relu(self.Linear1(input_data))
        out, hidden = self.rnn(out)
        #out1 is for onset & offset
        out1 = torch.sigmoid(self.Linear2(out))
        #out2 is for pitch
        out2 = self.Linear3(out)
        return out1, out2

class MyData(Data.Dataset):
    def __init__(self, data_seq, label):
        self.data_seq = data_seq
        self.label= label

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
            #'label': self.label[idx]
        }

def collate_fn(samples):
    batch = {}
    #print (samples[0]['data'].shape)
    temp= [torch.from_numpy(np.array(sample['data'], dtype= np.float32)) for sample in samples]
    padded_data = rnn_utils.pad_sequence(temp, batch_first=True, padding_value= 0)
    batch['data']= padded_data
    #batch['label']= [np.array(sample['label'], dtype= np.float32) for sample in samples]

    return batch

with open("test_all.pkl", 'rb') as pkl_file:
    train_data= pickle.load(pkl_file)
BATCH_SIZE = 1
loader = Data.DataLoader(dataset=train_data, batch_size= BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print(torch.shape(loader))
for batch_idx, sample in enumerate(loader):
    print(batch_idx)
    print(sample)
    exit()