#!/home/kartik/SIH/pipeline/envs/py3/bin/python
#General imports
import numpy as np
import pandas as pd
#Parse Imports
import subprocess
#Feature Extraction Imports
import joblib
from sliding_window_processor import collect_event_ids
#Model imports
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

#Parsing
interpreter_c = "./envs/py2/bin/python" #! CHANGE!!
python_script_b = "pipeline_support.py"
subprocess.run([interpreter_c, python_script_b])

#Feature Extraction
data_version = "_v5"
data_version = "_tf-idf{}".format(data_version)
load_data_location = "./dataset/"
save_location = "./dataset/"
print("Loading x_test")
x_test = pd.read_csv("{}HDFS.log_structured.csv".format(load_data_location)) #! CHANGE!!
re_pat = r"(blk_-?\d+)"
col_names = ["BlockId", "EventSequence"]
print("Collecting events for features")
events_test = collect_event_ids(x_test, re_pat, col_names)
events_test_values = events_test["EventSequence"].values
loaded_fe = joblib.load('fe_model.pkl')
print("Transforming features")
subblocks_test = loaded_fe.transform(events_test_values)
np.save("{}x_test{}.npy".format(save_location, data_version), subblocks_test)

#Prediction
random.seed(0)
np.random.seed(0)
data_loc = "./dataset/"
test_data = np.load('{}x_test_tf-idf_v5.npy'.format(data_loc))

class logDataset(Dataset):
    """Log Anomaly Features Dataset"""
    
    def __init__(self, data_vec, labels=None):
        self.X = data_vec
        self.y = labels
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        data_matrix = self.X[idx]
        
        if not self.y is None:
            return(data_matrix, self.y[idx])
        else:
            return data_matrix

test_data = torch.tensor(test_data, dtype=torch.float32)
test_data = F.pad(input=test_data, pad=(1, 1, 1, 1), mode='constant', value=0)
test_data = np.expand_dims(test_data, axis=1)
BATCH_SIZE = 128
NUM_CLASSES = 2
test_dataset = logDataset(test_data)
test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=0,
                         shuffle=False)

class logCNN(nn.Module):

    def __init__(self, num_classes):
        super(logCNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            
            nn.Conv2d(1, 16, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(

            nn.Linear(1056, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
model = logCNN(NUM_CLASSES)
model.load_state_dict(torch.load('model.pth'))

# Evaluate metrics on the test set
y_hats = []
for i, inputs in enumerate(test_loader):
    yhat = model(inputs)[-1].cpu().detach().numpy().round()
    yhat = np.argmax(yhat, axis=1)
    y_hats.append(yhat)
y_hats = [item for sublist in y_hats for item in sublist]

np.savetxt("dataset/predictions.csv",
        y_hats,
        delimiter =", ",
        fmt ='% s')
print('Done.')
#? Storing prediction and logs

#? Deleting dataset
