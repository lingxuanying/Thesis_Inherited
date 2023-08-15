# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:41:59 2023

@author: iir
"""





import pandas as pd
import numpy as np
import random
import sys
import pickle
import math
from numpy.linalg import norm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
from math import e
import warnings
warnings.filterwarnings("ignore")

def cal_time_step(data):
    date = data['start-analysis-date'][0]
    record_num = 0 # calculate同次透析第幾筆紀錄
    time_s = []
    time = []
    
    for i in range(len(data)):
        # 若屬於同一筆透析資料
        if (data['start-analysis-date'][i] == date) : 
            record_num += 1
    
        else:
            date = data['start-analysis-date'][i]
            record_num = 0
            # 當該次透析跑完開始計算時間差
            for j in range(len(time_s)):
                time.append(240 - (time_s[j] - time_s[0]).seconds / 60)
            time_s = []
        time_s.append(data['start-time'][i])
    
    for j in range(len(time_s)):
        time.append(240 - (time_s[j] - time_s[0]).seconds / 60)
    data['time_step'] = time

def generate_pickle(train_data, train_list, type_name):
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

    temp = list(train_data[temperature][i[0]:i[0]+1].values[0])
    temp = [-1 if math.isnan(x) else x for x in temp]
    temperature_list.append(temp)
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
  f = open(type_name + '_file.pickle','wb')
  pickle.dump(traindata,f)
  
def generate_pickle_without_temperature(train_data, train_list):
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

def normalize(data):
  df = pd.DataFrame(data)
  x = df.values #returns a numpy array
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df = pd.DataFrame(x_scaled)
  temperature_list = df.values
  return temperature_list

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


def adjust_input(input, zero_list):
  for i in range(len(input)):
    while(len(input[i]) < record_num):
      input[i].append(zero_list)
  return np.array(input)

def zero_norm(data):
  mean, std, var = torch.mean(data), torch.std(data), torch.var(data)
  data = (data-mean)/std
  return data

def cal_x_len(data):
  l = []
  for i in data:
    l.append(len(i))
  return np.array(l)

#%% GRU function
Dense = torch.nn.Linear
LayerNorm = torch.nn.LayerNorm
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        reshaped_input = input_seq.contihuous().view(-1, input_seq.size(-1))
        output = self.module(reshaped_input)

        if self.batch_first:
            output = output.contihuous().view(input_seq.siez(0), -1, output.size(-1))
        else:
            output = output.contihuous().view(-1, input_seq.siez(0), output.size(-1))
        return output

def linear_layer(input_size, size, activation=None, use_time_distributed=False, use_bias=True):
    linear = torch.nn.Linear(input_size, size, bias=use_bias)
    if use_time_distributed:
        linear = TimeDistributed(linear)
    return linear

def apply_gating_layer(x, hidden_layer_size, dropout_rate=None, use_time_distributed=True, activation=None):
    if dropout_rate is not None:
        x = torch.nn.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = TimeDistributed(
            torch.nn.Linear(x.shape[-1], hidden_layer_size))(
            x)
        gated_layer = TimeDistributed(
            torch.nn.Linear(x.shape[-1], hidden_layer_size))(
            x)
    else:
        activation_layer = torch.nn.Linear(
            x.shape[-1], hidden_layer_size)(
            x)
        gated_layer = torch.nn.Linear(
            x.shape[-1], hidden_layer_size)(
            x)

    return torch.mul(activation_layer, gated_layer), gated_layer

def add_and_norm(x, y):
    tmp = x + y
    tmp = LayerNorm(tmp.shape)(tmp)
    return tmp

def gated_residual_network(x, hidden_layer_size, output_size=None, dropout_rate=None, use_time_distributed=True, additional_context=None, return_gate=False):
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(x.shape[-1], output_size)
        if use_time_distributed:
            linear = TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        x.shape[-1],
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        x)

    hidden = torch.nn.ELU()(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
        return add_and_norm(skip, gating_layer), gate
    else:
        # print('skip: ', skip.shape)
        # print('gating_layer: ', gating_layer.shape)
        # print('add_and_norm(skip, gating_layer)', add_and_norm(skip, gating_layer).shape)
        return add_and_norm(skip, gating_layer)

def static_combine_and_mask(embedding):
    # Add temporal features
    _, num_time, num_static = embedding.shape

    flatten = torch.nn.Flatten()(embedding)

    # Nonlinear transformation with gated residual network.
    mlp_outputs = gated_residual_network(
        flatten,
        hidden_layer_size=5,
        output_size=num_static,
        dropout_rate=0.2,
        use_time_distributed=False,
        additional_context=None)
    sparse_weights = torch.nn.Softmax()(mlp_outputs)
    sparse_weights = torch.unsqueeze(sparse_weights, -1) # (24, 9, 1) 非時序特徵權重

    trans_emb_list = torch.tensor([])
    for i in range(num_static):
        e = gated_residual_network(
            embedding[:, :, i:i + 1],
            hidden_layer_size=1,
            dropout_rate=0.2,
            use_time_distributed=False)
        trans_emb_list = torch.cat((trans_emb_list, e), 2)

    transformed_embedding = torch.permute(trans_emb_list, (0, 2, 1))
    combined = torch.mul(sparse_weights, transformed_embedding)
    combined = torch.permute(combined, (0, 2, 1))
    static_vec = torch.sum(combined, 1)

    return combined, sparse_weights
class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers


        self.gru = nn.GRU(9, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(p=0.5)

        self.layer_t = nn.Linear(record_num, hidden_dim)
        self.layer_c = nn.Linear(input_dim, input_dim, bias=False)
        self.layer_adjust = nn.Linear(hidden_dim, hidden_dim)
        self.gru_adjust = nn.GRU(hidden_dim + 35, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.layer_final = nn.Linear(hidden_dim, 24)
        self.layer_combine = nn.Linear(24, 1)
        self.softmax = nn.Softmax(dim=0)

        self.layer_info = nn.Linear(9, 18)
        self.temp_encoder = nn.TransformerEncoderLayer(d_model=24, nhead=6)
        self.transformer_encoder = nn.TransformerEncoder(self.temp_encoder, num_layers=1)
        self.layer_temp = nn.Linear(24, 1)
        self.layer_upsample = nn.Linear(9, 9)

        #self.gru_temp = nn.GRU(batch_size, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)



    def forward(self, x, h, c, arpha_h, final_h, t, l, info, temp):
        decay_t = 1 / torch.log(e + t)
        decay_t = self.sigmoid(decay_t)

        #final_h = final_h * self.layer_t(decay_t)

        #print(decay_t.shape)
        #decay_temp = 1 / torch.log(27 - temp[:, 22])

        # unreliablility-aware attention
        out = self.batch_norm(torch.permute(x, (0, 2, 1)))
        out = torch.permute(out, (0, 2, 1))
        #arpha = self.sigmoid(c)
        #arpha = self.layer_upsample(c)
        out = self.layer_upsample(out)
        #unreliable_attention = arpha * out

        # Symptom-aware attention
        #c_01 = torch.floor(c)
        #c_01 = self.layer_upsample(c_01)
        decay_t = torch.unsqueeze(decay_t, 2)
        x_deep, h_deep = self.gru(out * decay_t, h) #左T-GRU


        #print(arpha_h.shape)
        #print(arpha.shape)

        #print(decay_t.shape)
        arpha_deep, arpha_h_deep = self.gru(decay_t * c, arpha_h) # 右T-GRU
        symptom_attention = x_deep * arpha_deep

        x_adjust = self.relu(self.layer_adjust(symptom_attention))

        # 將non-sequential由(24, 9)變為(24, 4, 9)與x_adjust(24, 4, 256)接在一起過最後的GRU

        info = info.unsqueeze(1)
        info = torch.cat((info, info, info, info), 1)
        static_encoder, static_weights = static_combine_and_mask(info)

        #print(temp[:, 22])
        #temp_x = torch.log(((27 - temp[:, 22])/4) ** 1.273)
        temp = self.transformer_encoder(temp)
        temp = torch.unsqueeze(temp, 0)
        temp = self.layer_temp(temp)

        #print(temp.shape)
        final_h = temp * final_h

        #temp = temp.unsqueeze(1)
        #temp = torch.cat((temp, temp, temp, temp), 1)
        #temp, temp_h = self.gru_temp(temp, temp_h)
        out, h_out = self.gru_adjust(torch.cat((x_adjust, static_encoder), 2), final_h)
        out = self.dropout(out)


        # Combine all the features
        out = self.layer_final(out[:,-1])
        f = self.sigmoid(self.layer_combine(out)) # 36->1

        return f, h_deep, arpha_h_deep, h_out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
    
def evaluate(model, test_x, test_u, test_t, test_y, test_l, test_info, test_temp):
    model.eval()
    outputs = []
    targets = []

    inp = torch.from_numpy(test_x)
    u_inp = torch.from_numpy(1+test_u)
    t_inp = torch.from_numpy(test_t)
    labs = torch.from_numpy(test_y)
    l_inp = torch.from_numpy(test_l)
    info_inp = torch.from_numpy(test_info)
    temp_inp = torch.from_numpy(test_temp)

    h = model.init_hidden(inp.shape[0])
    arpha_h = model.init_hidden(inp.shape[0])
    final_h = model.init_hidden(inp.shape[0])

    out, h, arpha_h, final_h = model(inp.to(device).float(), h, u_inp.to(device).float(), arpha_h, final_h, t_inp.to(device).float(), l_inp, info_inp.to(device).float(), temp_inp.to(device).float())
    print('Prediction Result: ' , end='')
    print( out.tolist())
    out = out>=0.5
    targets.append((labs.numpy()).reshape(-1))

    print('Label: ' , end='')
    print( targets[0])
    return outputs
#%% Parameter
criteria = sys.argv[1] # 90/40/MAP
history_num = int(sys.argv[2]) #利用最近5次發生過IDH的透析算相似度
data_file = sys.argv[3]
data_row_index_path = sys.argv[4]
pickle_path = sys.argv[5]
pickle_similarity_path = sys.argv[6]
gru_path = sys.argv[7] # The model of pretext task


#%%
print('Generate Pickle...')

file_name = data_file #'/content/drive/My Drive/IDH_Thesis/20220905新版資料/Training_data_篩過的.csv'
data = pd.read_csv(file_name, encoding='utf-8', engine='python')

data['start-time'] = pd.to_datetime(data['start-time'])
data['start-analysis-date'] = pd.to_datetime(data['start-analysis-date'])
data = data.replace('None',np.nan)
data['SP-start'] = pd.to_numeric(data['SP-start'], errors='coerce')

sequential = ['SP-start', 'DP-start', 'Dialysis-blood-rate', 'Dehydration-rate', 'HR', 'RR', 'blood-speed', 'normal_saline', 'Dialysis-blood-temp']
non_sequential = ['start-weight', 'predict-Dehydration', 'Sum_HTN_Drugs', 'DM', 'HTN', 'CAD', 'Age', 'Sex', 'frequency_'+criteria, 'Hb', 'Hct', 'MCV', 'RBC', 'WBC', 'Plt', 'CREA', 'eGFR', 'NA',
          'K', 'CA', 'P', 'CO2', 'URIC', 'BUN_前', 'BUN_後', 'Ferritin', 'TIBC', 'IRON', 'PTHi.', '8202CRP', 'HbA1C', 'LDL-C', 'CHOL', 'TRIG', '8203CRP']

temperature = []
for i in range(24, 48):
  temperature.append('temperature' + str(i))

from sklearn import preprocessing

for i in sequential + non_sequential + temperature:
  data[i] = pd.to_numeric(data[i],errors='coerce')

data = data.fillna(-1)
test_PatientID = pd.unique(data['ID'])
test_data = data[data["ID"].isin(test_PatientID)].reset_index(drop=True)
f = open(data_row_index_path, "r")
text = f.readlines()
info = str(text[0]).strip("\n").strip("'")
exec(info)
cal_time_step(test_data)
generate_pickle(test_data, test_list, pickle_path+'test')
print('Generate Pickle Successfully.')
#%%
print('Generate Pickle of Similarity...')
history_test = get_history(test_data, test_list)
test = generate_pickle_without_temperature(test_data, test_list)

# 計算similarity
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

similarity_test = cal_similarity(test, history_test)
similarity_test = normalize(similarity_test)
testing_file = pickle_path+'test_file.pickle'
test = pickle.load(open(testing_file, 'rb'))
u_test = generate_u(test, similarity_test)
f = open(pickle_similarity_path+'test_file.pickle','wb')
pickle.dump(u_test,f)
f = open(pickle_similarity_path+'test_file.pickle','wb')
pickle.dump(u_test,f)
print('Generate Pickle of Similarity Successfully.')
#%%
print('Start Predicting...')
batch_size = 24
criteria = sys.argv[1] # 90/40/MAP

testing_file = sys.argv[5]+'test_file.pickle'
output_file = sys.argv[8]

test = pickle.load(open(testing_file, 'rb'))

zero_list = -1
info_test = adjust_input(test[2], zero_list)
info_test = np.array(zero_norm(torch.from_numpy(info_test)))

# temperature
temp_test = adjust_input(test[4], zero_list)
zero_list = [-1] * feature_num

# feature's length (for last embedding)
l_test = cal_x_len(test[0])

# sequential variable
x_test = adjust_input(test[0], zero_list)

_, _, input_dimension = x_test.shape

# time step
zero_list = 480

t_test = adjust_input(test[3], zero_list)
t_test = np.array(zero_norm(torch.from_numpy(t_test)))

u_testing_file = sys.argv[6]+'test_file.pickle'
u_test = pickle.load(open(u_testing_file, 'rb'))

# unreliable
one_list = [0] * feature_num
u_test = adjust_input(u_test, one_list)

# label
y_test = np.array(test[1]) # (170, )
y_test = np.expand_dims(y_test, axis=1)
#%%
seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
#%%
lr = 0.001
accuracy_l = []
accuracy = 0

input_dim = feature_num
hidden_dim = 256
output_dim = 1
n_layers = 1
model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
model.to(device)
model.load_state_dict(torch.load(output_file))
gru_output = evaluate(model, x_test, u_test, t_test, y_test, l_test, info_test, temp_test)
