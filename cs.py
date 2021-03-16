import pickle
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

class MyData(Data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq
        #self.label= label

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
    batch['label']= [np.array(sample['label'], dtype= np.float32) for sample in samples]

    return batch

with open("testing.pkl", 'rb') as pkl_file:
    train_data= pickle.load(pkl_file)
BATCH_SIZE = 1
loader = Data.DataLoader(dataset=train_data, batch_size= BATCH_SIZE, shuffle=True)

for batch_idx, sample in enumerate(loader):
    print(batch_idx)
    print(len(sample))