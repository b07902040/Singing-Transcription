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
    def __init__(self, input_dim, hidden_size= 50):
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
def post_processing(output1, pitch):
    pitch= pitch.squeeze(1).squeeze(1).cpu().detach().numpy()
    #print (pitch.shape)
    #print (torch.mean(output1))
    threshold= 0.1
    notes= []
    this_onset= None
    this_offset= None
    this_pitch= None
    for i in range(len(output1)):
        if output1[i][0][0] > threshold and this_onset == None:
            this_onset= i
        elif output1[i][0][1] > threshold and this_onset != None and this_onset+ 1 < i and this_offset == None:
            this_offset= i
            this_pitch= int(round(np.mean(pitch[this_onset:this_offset+ 1])))
            notes.append([this_onset* 0.032+ 0.016, this_offset* 0.032+ 0.016, this_pitch])
            this_onset= None
            this_offset= None
            this_pitch= None
    #print (np.array(notes))
    return notes
def testing(net, sample, device):
    net.eval()
    data = sample['data']
    data = torch.Tensor(data)
    data = data.unsqueeze(1)
    #print (data.shape)
    data = data.to(device, dtype=torch.float)
    output1, output2 = net(data)
    #print (output1.shape)
    #print (output2.shape)
    answer = post_processing(output1, output2)
    #print(answer)
    #print(np.asarray(answer).shape)
    return answer

if __name__ == '__main__':
    print('Loading data ...')
    train_data= None
    with open("test_all.pkl", 'rb') as pkl_file:
        test_data= pickle.load(pkl_file)
    input_dim= 23
    hidden_size= 50
    BATCH_SIZE= 1
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        device = 'cpu'
    print("use",device,"now!")
    #for testing
    model = Myrnn(input_dim, hidden_size)
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load("ST_8.pt"))
    model.cuda()
    output = []
    for i in range(1500):
        if i%50 is 0:
            print(i)
        aut = testing(model, test_data[i], device)
        output.append(aut)
    output = np.asarray(output)
    print(output.shape)
    np.save('predict_all.npy', output)  