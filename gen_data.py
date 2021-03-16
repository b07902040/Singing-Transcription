import json
import os
import numpy as np
import pickle
import sys
import time
import torch.utils.data as Data

class MyData(Data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return {
            'data': self.data_seq[idx],
        }

THE_FOLDER = "testing"
data_seq= []
label= []
cnt = 0
for the_dir in os.listdir(THE_FOLDER):
    cnt += 1
    print(cnt)
    #print (the_dir)
    if not os.path.isdir(THE_FOLDER + "/" + the_dir):
        continue

    json_path = THE_FOLDER + "/" + the_dir+ f"/{the_dir}_feature.json"
    #gt_path= THE_FOLDER+ "/" +the_dir+ "/"+ the_dir+ "_groundtruth.txt"
    youtube_link_path= THE_FOLDER+ "/" + the_dir+ "/"+ the_dir+ "_link.txt"

    with open(json_path, 'r') as json_file:
        temp = json.loads(json_file.read())

    #gtdata = np.loadtxt(gt_path)

    data= []
    for key, value in temp.items():
        data.append(value)

    data= np.array(data).T

    data_seq.append(data)
    #label.append(gtdata)

testing_data = MyData(data_seq)

with open("testing.pkl", 'wb') as pkl_file:
    pickle.dump(testing_data, pkl_file)