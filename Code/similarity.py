# -*- coding: utf-8 -*-
"""計算透析歷史正樣本與當前的相似度_LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AZTNEUlE7pXp-TsloFXMnaTntQStU3TC
"""


import pandas as pd
import numpy as np
import random
import sys
pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")

print('Generate Pickle of Similarity')

criteria = sys.argv[1] # 90/40/MAP
history_num = int(sys.argv[4]) #利用最近5次發生過IDH的透析算相似度
data_file = sys.argv[5]
patient_id_path = sys.argv[6]
data_row_index_path = sys.argv[7]
pickle_load_path = sys.argv[8]
pickle_save_path = sys.argv[9]
gru_path = sys.argv[10] # The model of pretext task
#%%
file_name = data_file
data = pd.read_csv(file_name, encoding='utf-8', engine='python')

data['start-time'] = pd.to_datetime(data['start-time'])
data['start-analysis-date'] = pd.to_datetime(data['start-analysis-date'])
data['UF'] = pd.to_numeric(data['UF'],errors='coerce')
data = data.replace('None', np.nan)
data = data.replace(-1, np.nan)
data['SP-start'] = pd.to_numeric(data['SP-start'], errors='coerce')
data['SP-end'] = pd.to_numeric(data['SP-end'], errors='coerce')

sample_num = 5
sequential = ['SP-start', 'DP-start', 'Dialysis-blood-rate', 'Dehydration-rate', 'HR', 'RR', 'blood-speed', 'normal_saline', 'Dialysis-blood-temp']
#sequential = ['SP-start', 'UF', 'Age']
#non_sequential = ['start-weight', 'UF', 'temp-difference', 'sex', 'hypertension', 'cardiovascular', 'diabetes', 'Age', 'ACEi_ARB', 'Beta_Blockers', 'Alpha_Blockers', 'Non_DHP_CCB', 'DHP_CCB', 'Vasodilators', 'Nitrates', 'Sum_HTN_Drugs', 'frequency']
non_sequential = []

temperature = []
for i in range(24, 48):
  temperature.append('temperature' + str(i))
train_size = 0.8
valid_size = 0.1
test_size = 0.1

def normalize(data):
  df = pd.DataFrame(data)
  x = df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df = pd.DataFrame(x_scaled)
  temperature_list = df.values
  return temperature_list

from sklearn import preprocessing

for i in sequential:
  data[i] = pd.to_numeric(data[i],errors='coerce')
  data[i] = normalize(data[i]) # 對feature整欄做normalize

for i in non_sequential:
  data[i] = pd.to_numeric(data[i],errors='coerce')
  data[i] = normalize(data[i])

#data = data.fillna(-1)


"""# 從 Patient_ID.txt 直接讀 Train, Valid, Test"""

f = open(patient_id_path, "r")
#f = open("/content/drive/My Drive/IDH_Thesis/測試用/Patient_ID.txt", "r")
text = f.readlines()
for i in range(3):
  info = str(text[i]).strip("\n").strip("'")
  exec(info)

"""# 從 row_index.txt 直接讀 Train_list, Valid_list, Test_list"""

train_data = data[data["ID"].isin(train_PatientID)].reset_index(drop=True)
valid_data = data[data["ID"].isin(valid_PatientID)].reset_index(drop=True)
test_data = data[data["ID"].isin(test_PatientID)].reset_index(drop=True)


#f = open("/content/drive/My Drive/IDH_Thesis/20220729/split/20230323row_index"+criteria+".txt", "r")
f = open(data_row_index_path, "r")
text = f.readlines()
for i in range(3):
  info = str(text[i]).strip("\n").strip("'")
  exec(info)

"""# 找出最近五筆發生IDH紀錄"""

def get_history(train_data, train_list):
  num = 0
  history = []
  for i in train_list:
    PatientID = train_data['ID'][i[0]]
    filter = ((train_data['ID'] == PatientID) & (train_data['label_'+criteria] == True) & (train_data.index < i[0]))
    positive_df = train_data[sequential+["start-analysis-date"]][filter]
    positive_df = positive_df.drop_duplicates(subset=['start-analysis-date'])
    positive_record = positive_df['start-analysis-date'][-history_num:].tolist()
    history_s = []
    for j in positive_record:
      filter = ((train_data['ID'] == PatientID) & (train_data['start-analysis-date'] == j))
      history_s.append(train_data[sequential][filter][:4].values.tolist())
    history.append(history_s)
    num += 1
  return history #(n筆資料, 5筆history dialysis, 1~4筆紀錄, 9個feature)

s = sequential+["start-analysis-date"]

history_train = get_history(train_data, train_list)
history_valid = get_history(valid_data, valid_list)
history_test = get_history(test_data, test_list)
history_train

def generate_pickle(train_data, train_list):
  label = []
  sequential_list = []
  non_sequential_list = []
  time_step = []
  temperature_list = []
  for i in train_list:
    label.append(i[2])
    temp = list(train_data[non_sequential][i[0]:i[0]+1].values[0])
    temp = [-1 if math.isnan(x) else x for x in temp]
    non_sequential_list.append(temp)

    temp1 = []
    #temp2 = []
    temp3 = []
    for j in range(i[0], i[1]+1):
      temp = list(train_data[sequential][j:j+1].values[0])
      temp = [-1 if math.isnan(x) else x for x in temp]
      temp1.append(temp)
      #temp2.append(list(train_data[non_sequential][j:j+1].values[0]))
      temp3.append(train_data['time_step'][j:j+1].values[0])
    #print(len(temp2))
    sequential_list.append(temp1)
    time_step.append(temp3)
  traindata = []

  traindata.append(sequential_list)
  traindata.append(label)
  traindata.append(non_sequential_list)
  traindata.append(time_step)
  traindata.append(temperature_list)
  return traindata

train_data['time_step'][0:1].values[0]

import pickle
import math

#generate_pickle(train_data, unbalanced_train_list, '/content/drive/MyDrive/Temperature/HiTANet/data/hf_sample/model_inputs/20220729_try1_unbalanced/unbalanced/train')
#generate_pickle(valid_data, unbalanced_valid_list, '/content/drive/MyDrive/Temperature/HiTANet/data/hf_sample/model_inputs/20220729_try1_unbalanced/unbalanced/valid')
#generate_pickle(test_data, unbalanced_test_list, '/content/drive/MyDrive/Temperature/HiTANet/data/hf_sample/model_inputs/20220729_try1_unbalanced/unbalanced/test')

train = generate_pickle(train_data, train_list)
valid = generate_pickle(valid_data, valid_list)
test = generate_pickle(test_data, test_list)

"""# 計算相似度"""

from numpy.linalg import norm
import torch
import torch.nn as nn

class GRUNet_pretext(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super(GRUNet_pretext, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.layer_norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(p=0.5)

        self.layer_1 = nn.Linear(hidden_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 1)


    def forward(self, x, h):
        out = self.batch_norm(torch.permute(x, (0, 2, 1)))
        out = torch.permute(out, (0, 2, 1))
        out, h_out = self.gru(out, h)

        out = self.layer_1(out[:,-1])
        out = self.layer_norm(out)
        out = self.relu(out)

        out = self.layer_2(out)
        out = self.relu(out)


        return out, h_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

feature_num = 9
record_num = 4
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def data_preprocess(array):
  zero_list = [-1] * feature_num


  for i in range(len(array)):
      while(len(array[i]) < record_num):
          array[i].append(zero_list)
  x_train = np.array(array)
  return x_train

hidden_dim = 256
input_dim = 9
n_layers =  1
output_file = gru_path
model = GRUNet_pretext(input_dim, hidden_dim, n_layers)
model.load_state_dict(torch.load(output_file))

model.eval()
#inp = torch.from_numpy(x_test)
#h = model.init_hidden(inp.shape[0])

#out, h = model(inp.to(device).float(), h)
#print(out)

def cal_similarity(train, history):
  cos = nn.CosineSimilarity(dim=0)
  similarity_list = []
  history_small_than_current = 0 #雖然history都是全取，train是1-4 random，但還是會發生就算全取也小於train的數字，因此無法計算similarity需扣掉，僅有兩筆，都在train set裡
  for i in range(len(train[0])):
    temp = []
    for j in range(len(train[0][i])): # 一次紀錄9個feature
      similarity = 0
      inp = torch.from_numpy(data_preprocess([train[0][i][:j+1]]))
      inp = inp.masked_fill(torch.isnan(inp), -1)
      h = model.init_hidden(inp.shape[0])
      out, h = model(inp.to(device).float(), h)

      history_small_than_current = 0
      for k in range(len(history[i])):
        if(len(history[i][k][:j+1])==j+1): #若history的數量夠的話
          inp = torch.from_numpy(data_preprocess([history[i][k][:j+1]]))
          inp = inp.masked_fill(torch.isnan(inp), -1)
          h = model.init_hidden(inp.shape[0])
          history_out, h = model(inp.to(device).float(), h)
          #similarity = similarity - norm(torch.sub(out, history_out).cpu().detach().numpy())
          similarity = similarity + torch.mean(cos(out, history_out)).cpu().detach().numpy()

        else:
          history_small_than_current += 1
      if(len(history[i]) - history_small_than_current>0):
        similarity = similarity / (len(history[i]) - history_small_than_current)
      else:
        similarity = np.nan


      temp.append(similarity)
    similarity_list.append(temp)
  return similarity_list

similarity_train = cal_similarity(train, history_train)
similarity_valid = cal_similarity(valid, history_valid)
similarity_test = cal_similarity(test, history_test)

def normalize(data):
  df = pd.DataFrame(data)
  x = df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df = pd.DataFrame(x_scaled)
  temperature_list = df.values
  return temperature_list

similarity_train = normalize(similarity_train)
similarity_valid = normalize(similarity_valid)
similarity_test = normalize(similarity_test)

# training_file = '/content/drive/My Drive/IDH_Thesis//label_data_'+criteria+'/train_file.pickle'
# validation_file = '/content/drive/My Drive/IDH_Thesis/Training Data/label_data_'+criteria+'/valid_file.pickle'
# testing_file = '/content/drive/My Drive/IDH_Thesis/Training Data/label_data_'+criteria+'/test_file.pickle'
#testing_file = '/content/drive/My Drive/IDH_Thesis/Training Data/case study/test_file.pickle'

training_file = pickle_load_path+'train_file.pickle'
validation_file = pickle_load_path+'valid_file.pickle'
testing_file = pickle_load_path+'test_file.pickle'

train = pickle.load(open(training_file, 'rb'))
validate = pickle.load(open(validation_file, 'rb'))
test = pickle.load(open(testing_file, 'rb'))

def generate_u(train, similarity_train):
  feature_num = 9
  u_train = []
  for i in range(len(train[0])):
    u_train_s = []
    for j in range(len(train[0][i])):
      if(math.isnan(similarity_train[i][j])):
        u_train_s.append([float(-1)] * feature_num)
      else:
        u_train_s.append([float(similarity_train[i][j])] * feature_num)
    u_train.append(u_train_s)
  return u_train

u_train = generate_u(train, similarity_train)
u_valid = generate_u(validate, similarity_valid)
u_test = generate_u(test, similarity_test)

f = open(pickle_save_path+'train_file.pickle','wb')
pickle.dump(u_train,f)

f = open(pickle_save_path+'valid_file.pickle','wb')
pickle.dump(u_valid,f)

f = open(pickle_save_path+'test_file.pickle','wb')
pickle.dump(u_test,f)

#f = open('/content/drive/My Drive/IDH_Thesis/Training Data/new_similarity/similarity_lstm_40_case_study/test_file.pickle','wb')
f = open(pickle_save_path+'test_file.pickle','wb')

pickle.dump(u_test,f)

print('Generate Pickle of Similarity Successfully.')