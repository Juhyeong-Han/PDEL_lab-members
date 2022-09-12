# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import numbers
from sklearn.metrics import f1_score
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa

# 피해도(class 1을 많게, class 0을 적게)
df = pd.read_csv('벼예찰포 통합자료.csv')
df['조사병해충'].unique()
# 잎집무늬마름병(병든줄기율),잎집무늬마름병(수직진전도),잎집무늬마름병(피해도)


df = df.loc[(df['조사병해충'] == '잎집무늬마름병(피해도)') & (df['조사년도'] >= 2002)].drop('조사병해충',axis =1)    # 2002년 이후 도열병데이터만 출력
peak_df = df.loc[~df['peak'].isna(), ['경도','위도','시도','시군구','조사년도','peak','audpc']]                   # 'peak'가 na가 아닌 것들의 지정 컬럼만 출력
peak_df_size = peak_df.iloc[0:1500,:]

peak_df_size = peak_df_size.sort_values(['시도','조사년도','시군구'])

peak_df_size = peak_df_size.reset_index(drop = True)
#peak_df = peak_df.loc[peak_df['peak']>0]
#peak_df


# 기상 관측소마다 관측 시작/종료 일시가 다르기 때문에 맨 위, 맨 아래 행의 날짜를 읽어 Start year와 End year를 새로운 열에 저장",
# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 아래에서 구할 때 해당 연도가 Start year와 End year 사이에 있는지 확인할 것임",

station_df = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/Station-Info-all-129stns.csv').loc[27:]     #### 북한 제외
station_df = station_df.reset_index(drop=True)

def get_EYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/'+stnID + '.csv').iloc[-1,0:2]           # 관측소 ID를 이용, 가장 마지막 관측 년,월 출력, 월이 10월 미만일시 년 -1
    if mon >= 10:
        return year
    else:
        return year - 1

def get_SYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/' + stnID + '.csv').iloc[0,0:2]
    if mon <= 3:
        return year
    else:
        return year + 1
                                                                                                                         #### 파종과 추수의 시간
station_df['EYear'] = station_df['ID'].apply(lambda x: get_EYear(x))
station_df['SYear'] = station_df['ID'].apply(lambda x: get_SYear(x))

# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 구하는 과정",
# 로직이 복잡하니 가운데 괄호부터 실행해 보면서 어떻게 돌아가는지 파악하면 좋을듯",
from haversine import haversine               # pip install

peak_df_size['stnID'] = peak_df_size.apply(lambda county: 
                                 station_df.loc[station_df.apply(lambda station:
                                                                 haversine((county['위도'], county['경도']), (station['Lat'], station['Lon']), unit = 'km')
                                                                           if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                                           axis = 1).idxmin(), 'ID'],axis=1)

peak_df_size['dist'] = peak_df_size.apply(lambda county:
                                station_df.apply(lambda station:
                                                 haversine((county['위도'], county['경도']),(station['Lat'], station['Lon']), unit = 'km')
                                                 if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                 axis = 1).min(), axis = 1)  

    
    
    
    
peak_df_size = peak_df_size.loc[peak_df_size['dist'] <= 36]                        # 가장 가까운 관측소와 36km이상 떨어져 있는 경우 제외

# 그냥 분포 확인용
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'svg'
#pip install -U kaleido

px.histogram(peak_df, x='dist',nbins =100)

# Train and Test Model
# src 폴더의 ncpms.py 코드와 함께 볼 것
# input data를 설정하는 셀로, model_type = 'mother'일 때 Blast_FFNN 모델이 되며, months, variables, period는 필요 없으므로 None으로 두고 돌리면 됨",
# model_type = 'lstm', 'ffnn'일 때는 세 변수를 지정해 줘야 함",
# range(n, m)은 n부터 m - 1까지 iterate함을 의미",
# units와 activation function을 바꾸고 싶으면 design_model의 layer_info를 input으로 주면서 코드를 일부 수정해야 함. 현재는 최적값으로 세팅된 상태",
# months = range(5,8),
# variables = ['tmax', 'tmin', 'prec']
# period = 3

# Blast_FFNN모델 
months, variables, period = None, None, None
year_size = 1
model_type = 'mother' # ['mother', 'lstm', 'ffnn']

# 'ffnn' 모델
months, variables, period = [3,4,5,6,7], ['tmin','wspd','prec'], 16                 # 최적 조건
year_size = 1
model_type = 'ffnn'

# 'lstm' 모델
months, variables, period = [3,4,5,6,7], ['tmax','rhum','prec'],24                           # 최적조건
year_size = 3
model_type = 'lstm'


peak_df_size = peak_df_size.copy()

from src.ncpms import get_data, classify_disease_score, split_input_output, make_dataset, design_model, train_model

peak_df_size = get_data(peak_df_size, model_type, year_size, months, period, variables)
classify_disease_score(peak_df_size, year_size)
x, y = split_input_output(peak_df_size, model_type, period, year_size, variables)


from sklearn.model_selection import StratifiedKFold
from tqdm import notebook                                                         # 진행률을 알려주는 함수

kfold = StratifiedKFold(n_splits=10, shuffle=False)

predict_list = []
answer_list = []

for train_index, test_index in notebook.tqdm(kfold.split(x, y), total=10):
    train_x_disease_score, train_x_climate, test_x_disease_score, test_x_climate, train_y, test_y = make_dataset(x, y, model_type, train_index, test_index, year_size, period, variables)
    model = design_model(model_type, train_x_disease_score, train_x_climate)

    train_model(model, model_type, train_x_disease_score, train_x_climate, train_y)

    predict_list += list(model.predict([test_x_climate, test_x_disease_score] if model_type in ['lstm', 'ffnn']
                              else test_x_disease_score).argmax(axis=1))
    answer_list += list([int(a[-1]) for a in test_y.idxmax(axis=1).values])
    
    
test_df = pd.DataFrame({'predict':predict_list, 'actual':answer_list})
test_df['result'] = test_df.apply(lambda row: 'TP' if row['predict'] and row['actual']
                          else 'TN' if not row['predict'] and not row['actual']
                          else 'FN' if not row['predict'] and row['actual']
                          else 'FP' if row['predict'] and not row['actual']
                          else None, axis=1)

cm_df = pd.get_dummies(test_df['result']).sum()
TN, FN, FP, TP = cm_df.loc[['TN', 'FN', 'FP', 'TP']]
    
accuracy = (TP + TN) / (TP + FN + FP + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 / (1/precision + 1/recall)

print(f'dataset size : {len(test_df)}')
print(f'accuracy : {accuracy}')
print(f'precision : {precision}')
print(f'recall : {recall}')
print(f'f1_score : {f1_score}')

y_class = y=='class 0'
y_class.sum()
y_class==False

test_df.to_csv('피해도 optimal recall test_df_rank.csv', index=False)
x.to_csv('피해도 optimal recall X_rank.csv')
y.to_csv('피해도 optimal recall Y_rank.csv')


# =============================================================================
# =============================================================================
# =============================================================================
# 병든줄기율

df = pd.read_csv('벼예찰포 통합자료.csv')
df['조사병해충'].unique()
# 잎집무늬마름병(병든줄기율),잎집무늬마름병(수직진전도),잎집무늬마름병(피해도)


df = df.loc[(df['조사병해충'] == '잎집무늬마름병(병든줄기율)') & (df['조사년도'] >= 2002)].drop('조사병해충',axis =1)    # 2002년 이후 도열병데이터만 출력
peak_df = df.loc[~df['peak'].isna(), ['경도','위도','시도','시군구','조사년도','peak','audpc']]                   # 'peak'가 na가 아닌 것들의 지정 컬럼만 출력
peak_df_size = peak_df.iloc[0:2000,:]

peak_df_size = peak_df_size.sort_values(['시도','조사년도','시군구'])

peak_df_size = peak_df_size.reset_index(drop = True)
#peak_df = peak_df.loc[peak_df['peak']>0]
#peak_df


# 기상 관측소마다 관측 시작/종료 일시가 다르기 때문에 맨 위, 맨 아래 행의 날짜를 읽어 Start year와 End year를 새로운 열에 저장",
# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 아래에서 구할 때 해당 연도가 Start year와 End year 사이에 있는지 확인할 것임",

station_df = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/Station-Info-all-129stns.csv').loc[27:]     #### 북한 제외
station_df = station_df.reset_index(drop=True)

def get_EYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/'+stnID + '.csv').iloc[-1,0:2]           # 관측소 ID를 이용, 가장 마지막 관측 년,월 출력, 월이 10월 미만일시 년 -1
    if mon >= 10:
        return year
    else:
        return year - 1

def get_SYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/' + stnID + '.csv').iloc[0,0:2]
    if mon <= 3:
        return year
    else:
        return year + 1
                                                                                                                         #### 파종과 추수의 시간
station_df['EYear'] = station_df['ID'].apply(lambda x: get_EYear(x))
station_df['SYear'] = station_df['ID'].apply(lambda x: get_SYear(x))

# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 구하는 과정",
# 로직이 복잡하니 가운데 괄호부터 실행해 보면서 어떻게 돌아가는지 파악하면 좋을듯",
from haversine import haversine               # pip install

peak_df_size['stnID'] = peak_df_size.apply(lambda county: 
                                 station_df.loc[station_df.apply(lambda station:
                                                                 haversine((county['위도'], county['경도']), (station['Lat'], station['Lon']), unit = 'km')
                                                                           if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                                           axis = 1).idxmin(), 'ID'],axis=1)

peak_df_size['dist'] = peak_df_size.apply(lambda county:
                                station_df.apply(lambda station:
                                                 haversine((county['위도'], county['경도']),(station['Lat'], station['Lon']), unit = 'km')
                                                 if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                 axis = 1).min(), axis = 1)  

    
    
peak_df_size = peak_df_size.loc[peak_df_size['dist'] <= 36]                        # 가장 가까운 관측소와 36km이상 떨어져 있는 경우 제외

# 그냥 분포 확인용
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'svg'
#pip install -U kaleido

px.histogram(peak_df, x='dist',nbins =100)

# Train and Test Model
# src 폴더의 ncpms.py 코드와 함께 볼 것
# input data를 설정하는 셀로, model_type = 'mother'일 때 Blast_FFNN 모델이 되며, months, variables, period는 필요 없으므로 None으로 두고 돌리면 됨",
# model_type = 'lstm', 'ffnn'일 때는 세 변수를 지정해 줘야 함",
# range(n, m)은 n부터 m - 1까지 iterate함을 의미",
# units와 activation function을 바꾸고 싶으면 design_model의 layer_info를 input으로 주면서 코드를 일부 수정해야 함. 현재는 최적값으로 세팅된 상태",
# months = range(5,8),
# variables = ['tmax', 'tmin', 'prec']
# period = 3

# Blast_FFNN모델 
months, variables, period = None, None, None
year_size = 1
model_type = 'mother' # ['mother', 'lstm', 'ffnn']

# 'ffnn' 모델
months, variables, period = [3,4,5,6,7], ['tmax','wspd','rhum'], 15                 # 최적 조건
year_size = 1
model_type = 'ffnn'

# 'lstm' 모델
months, variables, period = [3,4,5,6,7], ['tmax','rhum','prec'],24                           # 최적조건
year_size = 3
model_type = 'lstm'


peak_df_size = peak_df_size.copy()

from src.ncpms import get_data, classify_disease_score, split_input_output, make_dataset, design_model, train_model

peak_df_size = get_data(peak_df_size, model_type, year_size, months, period, variables)
classify_disease_score(peak_df_size, year_size)
x, y = split_input_output(peak_df_size, model_type, period, year_size, variables)


from sklearn.model_selection import StratifiedKFold
from tqdm import notebook                                                         # 진행률을 알려주는 함수

kfold = StratifiedKFold(n_splits=10, shuffle=False)

predict_list = []
answer_list = []

for train_index, test_index in notebook.tqdm(kfold.split(x, y), total=10):
    train_x_disease_score, train_x_climate, test_x_disease_score, test_x_climate, train_y, test_y = make_dataset(x, y, model_type, train_index, test_index, year_size, period, variables)
    model = design_model(model_type, train_x_disease_score, train_x_climate)

    train_model(model, model_type, train_x_disease_score, train_x_climate, train_y)

    predict_list += list(model.predict([test_x_climate, test_x_disease_score] if model_type in ['lstm', 'ffnn']
                              else test_x_disease_score).argmax(axis=1))
    answer_list += list([int(a[-1]) for a in test_y.idxmax(axis=1).values])
    
    
test_df = pd.DataFrame({'predict':predict_list, 'actual':answer_list})
test_df['result'] = test_df.apply(lambda row: 'TP' if row['predict'] and row['actual']
                          else 'TN' if not row['predict'] and not row['actual']
                          else 'FN' if not row['predict'] and row['actual']
                          else 'FP' if row['predict'] and not row['actual']
                          else None, axis=1)

cm_df = pd.get_dummies(test_df['result']).sum()
TN, FN, FP, TP = cm_df.loc[['TN', 'FN', 'FP', 'TP']]
    
accuracy = (TP + TN) / (TP + FN + FP + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 / (1/precision + 1/recall)

print(f'dataset size : {len(test_df)}')
print(f'accuracy : {accuracy}')
print(f'precision : {precision}')
print(f'recall : {recall}')
print(f'f1_score : {f1_score}')

y_class = y=='class 0'
y_class.sum()
y_class==False



# =============================================================================
# =============================================================================
# =============================================================================
# 수직진전도

df = pd.read_csv('벼예찰포 통합자료.csv')
df['조사병해충'].unique()
# 잎집무늬마름병(병든줄기율),잎집무늬마름병(수직진전도),잎집무늬마름병(피해도)


df = df.loc[(df['조사병해충'] == '잎집무늬마름병(수직진전도)') & (df['조사년도'] >= 2002)].drop('조사병해충',axis =1)    # 2002년 이후 도열병데이터만 출력
peak_df = df.loc[~df['peak'].isna(), ['경도','위도','시도','시군구','조사년도','peak','audpc']]                   # 'peak'가 na가 아닌 것들의 지정 컬럼만 출력
peak_df_size = peak_df.iloc[0:2000,:]

peak_df_size = peak_df_size.sort_values(['시도','조사년도','시군구'])

peak_df_size = peak_df_size.reset_index(drop = True)
#peak_df = peak_df.loc[peak_df['peak']>0]
#peak_df


# 기상 관측소마다 관측 시작/종료 일시가 다르기 때문에 맨 위, 맨 아래 행의 날짜를 읽어 Start year와 End year를 새로운 열에 저장",
# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 아래에서 구할 때 해당 연도가 Start year와 End year 사이에 있는지 확인할 것임",

station_df = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/Station-Info-all-129stns.csv').loc[27:]     #### 북한 제외
station_df = station_df.reset_index(drop=True)

def get_EYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/'+stnID + '.csv').iloc[-1,0:2]           # 관측소 ID를 이용, 가장 마지막 관측 년,월 출력, 월이 10월 미만일시 년 -1
    if mon >= 10:
        return year
    else:
        return year - 1

def get_SYear(stnID):
    year, mon = pd.read_csv('C:/Users/B550-PRO V2/Desktop/NCPMS_ML/NCPMS_ML/data/User/' + stnID + '.csv').iloc[0,0:2]
    if mon <= 3:
        return year
    else:
        return year + 1
                                                                                                                         #### 파종과 추수의 시간
station_df['EYear'] = station_df['ID'].apply(lambda x: get_EYear(x))
station_df['SYear'] = station_df['ID'].apply(lambda x: get_SYear(x))

# NCPMS data 각 행에 대해 '가장 가까운 기상관측소'를 구하는 과정",
# 로직이 복잡하니 가운데 괄호부터 실행해 보면서 어떻게 돌아가는지 파악하면 좋을듯",
from haversine import haversine               # pip install

peak_df_size['stnID'] = peak_df_size.apply(lambda county: 
                                 station_df.loc[station_df.apply(lambda station:
                                                                 haversine((county['위도'], county['경도']), (station['Lat'], station['Lon']), unit = 'km')
                                                                           if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                                           axis = 1).idxmin(), 'ID'],axis=1)

peak_df_size['dist'] = peak_df_size.apply(lambda county:
                                station_df.apply(lambda station:
                                                 haversine((county['위도'], county['경도']),(station['Lat'], station['Lon']), unit = 'km')
                                                 if county['조사년도'] >= station['SYear'] and county['조사년도'] <= station['EYear'] else 999,
                                                 axis = 1).min(), axis = 1)  

 
peak_df_size = peak_df_size.loc[peak_df_size['dist'] <= 36]                        # 가장 가까운 관측소와 36km이상 떨어져 있는 경우 제외

# 그냥 분포 확인용
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'svg'
#pip install -U kaleido

px.histogram(peak_df, x='dist',nbins =100)

# Train and Test Model
# src 폴더의 ncpms.py 코드와 함께 볼 것
# input data를 설정하는 셀로, model_type = 'mother'일 때 Blast_FFNN 모델이 되며, months, variables, period는 필요 없으므로 None으로 두고 돌리면 됨",
# model_type = 'lstm', 'ffnn'일 때는 세 변수를 지정해 줘야 함",
# range(n, m)은 n부터 m - 1까지 iterate함을 의미",
# units와 activation function을 바꾸고 싶으면 design_model의 layer_info를 input으로 주면서 코드를 일부 수정해야 함. 현재는 최적값으로 세팅된 상태",
# months = range(5,8),
# variables = ['tmax', 'tmin', 'prec']
# period = 3

# Blast_FFNN모델 
months, variables, period = None, None, None
year_size = 1
model_type = 'mother' # ['mother', 'lstm', 'ffnn']

# 'ffnn' 모델
months, variables, period = [3,4,5,6,7], ['tmax','wspd','rhum'],21               # 최적 조건
year_size = 1
model_type = 'ffnn'

# 'lstm' 모델
months, variables, period = [3,4,5,6,7], ['tmax','rhum','prec'],24                           # 최적조건
year_size = 3
model_type = 'lstm'


peak_df_size = peak_df_size.copy()

from src.ncpms import get_data, classify_disease_score, split_input_output, make_dataset, design_model, train_model

peak_df_size = get_data(peak_df_size, model_type, year_size, months, period, variables)
classify_disease_score(peak_df_size, year_size)
x, y = split_input_output(peak_df_size, model_type, period, year_size, variables)


from sklearn.model_selection import StratifiedKFold
from tqdm import notebook                                                         # 진행률을 알려주는 함수

kfold = StratifiedKFold(n_splits=10, shuffle=False)

predict_list = []
answer_list = []

for train_index, test_index in notebook.tqdm(kfold.split(x, y), total=10):
    train_x_disease_score, train_x_climate, test_x_disease_score, test_x_climate, train_y, test_y = make_dataset(x, y, model_type, train_index, test_index, year_size, period, variables)
    model = design_model(model_type, train_x_disease_score, train_x_climate)

    train_model(model, model_type, train_x_disease_score, train_x_climate, train_y)

    predict_list += list(model.predict([test_x_climate, test_x_disease_score] if model_type in ['lstm', 'ffnn']
                              else test_x_disease_score).argmax(axis=1))
    answer_list += list([int(a[-1]) for a in test_y.idxmax(axis=1).values])
    
    
test_df = pd.DataFrame({'predict':predict_list, 'actual':answer_list})
test_df['result'] = test_df.apply(lambda row: 'TP' if row['predict'] and row['actual']
                          else 'TN' if not row['predict'] and not row['actual']
                          else 'FN' if not row['predict'] and row['actual']
                          else 'FP' if row['predict'] and not row['actual']
                          else None, axis=1)

cm_df = pd.get_dummies(test_df['result']).sum()
TN, FN, FP, TP = cm_df.loc[['TN', 'FN', 'FP', 'TP']]
    
accuracy = (TP + TN) / (TP + FN + FP + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 / (1/precision + 1/recall)

print(f'dataset size : {len(test_df)}')
print(f'accuracy : {accuracy}')
print(f'precision : {precision}')
print(f'recall : {recall}')
print(f'f1_score : {f1_score}')

y_class = y=='class 0'
y_class.sum()
y_class==False
