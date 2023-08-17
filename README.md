# README
## Reproduce the Performance of My Thesis / Training without Regenerating Pickles
`python <Reproduce Code> <Pickle Path> <Similarity Pickle Path> <Model Path>`


| Arguments      |   備註|
| -------------- | ------------------------------------------------------------- |
| Reproduce Code | Reproduce_nadir90.py / Reproduce_fall40.py / Reproduce_map.py |
| Pickle Path               |   Pickle儲存位置 |
| Similarity Pickle Path|  Similarity的Pickle儲存位置 |
|  Model Path  | 用來儲存模型的位置 |

### Eg. 
NADIR 90 (Systolic Blood Pressure (SBP) < 90 mmHg)
```
python Reproduce_nadir90.py ./Dataset/Training_Data/label_data_90/ ./Dataset/Training_Data/new_similarity/similarity_lstm_90/ ./Dataset/Training_Data/cache/gru
```
FALL 40 (Decrease in SBP by ≥ 40 mm Hg)
```
python Reproduce_fall40.py ./Dataset/Training_Data/label_data_40/ ./Dataset/Training_Data/new_similarity/similarity_lstm_40/ ./Dataset/Training_Data/cache/gru
```

FALL 20 (Decrease in SBP by ≥20 mm Hg or Decrease in MAP by ≥10 mm Hg associated with symptom)
```
python Reproduce_map.py ./Dataset/Training_Data/label_data_MAP/ ./Dataset/Training_Data/new_similarity/similarity_lstm_MAP/ ./Dataset/Training_Data/cache/gru
```
## Training with Regenerating Pickles
`python training_script.py <Criteria> <Re-split of Patients> <Reallocation of Data> <History Number> <Source File> <Save / Reload the Patients' ID> <Save / Reload the Data Row Index> <Pickle Path> <Similarity Pickle Path><Pretext Task Model Path><Model Path>`

| Arguments                        | 備註                                               |
| -------------------------------- | -------------------------------------------------- |
| Criteria                         | 90 / 40 / map                                      |
| Re-split of Patients             | (0/1)是否重新依照病患切分Train / Validation / Test |
| Reallocation of Data             | (0/1)是否重新抓取病患的正負樣本                    |
| History Number                   | 計算Simlarity時參考的過去發生IDH的數量             |
| Source File                      | Raw Data                                           |
| Save / Reload the Patients' ID   | 病患 Train/ Validation / Test 儲存位置             |
| Save / Reload the Data Row Index | 病患欲抓取的正負樣本 Row Index 儲存位置            |
| Pickle Path                      | Pickle儲存位置                                     |
| Similarity Pickle Path           | Similarity的Pickle儲存位置                         |
|   Pretext Task Model Path |  用來儲存預訓練模型的位置  |
| Model Path   | 用來儲存模型的位置          |


### Eg.  
```
python training_script.py 90 0 0 5 ./Dataset/Training_data.csv ./Dataset/Test/Patient_ID.txt ./Dataset/Test/20230323row_index_90.txt ./Dataset/Test/label_data_90/ ./Dataset/Test/similarity_lstm_90/ ./Dataset/Model_Preprocess/cache/gru ./Dataset/Training_Data/cache/gru
```


## Inference
`python prediction.py <Criteria> <History Number> <Inference Source File> <Row Index to Predict> <Pickle Path> <Similarity Pickle Path> <Pretext Task Model Path> <Model Weights>`
| Arguments                        | 備註                                               |
| -------------------------------- | -------------------------------------------------- |
| Criteria                         | 90 / 40 / map                                      |
| History Number                   | 計算Simlarity時參考的過去發生IDH的數量             |
| Inference Source File    | Raw Data    |
| Row Index to Predict | 欲預測的Row Index |
| Pickle Path                      | Pickle儲存位置                                     |
| Similarity Pickle Path           | Similarity的Pickle儲存位置                         |
|   Pretext Task Model Path |  用來儲存預訓練模型的位置  |
| Model Weights   | 模型Weights的位置          |
### Eg. 
```
python prediction.py 90 5 inference.csv inference.txt ./Dataset/Inference/label/ ./Dataset/Inference/similarity/ ./Dataset/Model_Preprocess/cache/gru ./Model/gru_90.zip
```