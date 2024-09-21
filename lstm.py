import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plott从 ing
import seaborn as sns  # 基于matplotlib数据可视化库
from scipy import stats
from scipy.stats import norm, skew
from sklearn.metrics import mean_squared_error  # RMSE均方根误差
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
scaler = MinMaxScaler()
results_df = pd.DataFrame(columns=['station-name', 'para'])
time_stamp = 12
files = os.listdir('/share/home/kong/github/CUG-hydro/HMs-zjw/data/ML-station1948-2014(gao)')
for i in range(0,15264):
    df = pd.read_csv('/share/home/kong/github/CUG-hydro/HMs-zjw/data/ML-station1948-2014(gao)/'+ files[i])
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
    train = train_data[['P', 'PET', 'Tem', 'SMelt','Qmm']]
    valid = test_data[['P', 'PET', 'Tem', 'SMelt','Qmm']]
    scaler = MinMaxScaler(feature_range=(0, 1))  
    scaled_data = scaler.fit_transform(train)
    x_train, y_train = [], []

    for j in range(time_stamp, len(train)):
        x_train.append(scaled_data[j - time_stamp:j,0:4])
        y_train.append(scaled_data[j, 4])


    x_train, y_train = np.array(x_train), np.array(y_train)

    scaled_data = scaler.fit_transform(valid)
    x_valid, y_valid = [], []
    for j in range(time_stamp, len(valid)):
        x_valid.append(scaled_data[j - time_stamp:j, 0:4])
        y_valid.append(scaled_data[j, 4])


    x_valid, y_valid = np.array(x_valid), np.array(y_valid)

    epochs = 500
    batch_size = 32

    def create_model(units=256, layers=2):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
        
        for _ in range(1, layers):
            model.add(LSTM(units=units))
        
        model.add(Dropout(0.01))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    model = KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

    param_grid = {
        'model__units': [64, 128, 256],
        'model__layers': [1, 2, 3]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(x_train, y_train)

    best_params = grid_result.best_params_
    results_df = results_df._append({'station-name': files[i], 'para': grid_result.best_params_}, ignore_index=True)
    model = create_model(layers=best_params['model__layers'],units=best_params['model__units'])
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])
    # 训练模型
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    R_pre1 = model.predict(x_valid)
    R_pre1 = R_pre1.flatten()
    R_pre1_reshaped = np.zeros((len(R_pre1), 5))  # 5 是原始特征的数量
    R_pre1_reshaped[:, 4] = R_pre1  # 将 y_valid 填充到最后一列
    R_pre1_inverse = scaler.inverse_transform(R_pre1_reshaped)
    R_pre1_final = R_pre1_inverse[:, 4]
    #R_pre1 = scaler.inverse_transform(R_pre1)
    y_valid_reshaped = np.zeros((len(y_valid), 5))  # 5 是原始特征的数量
    y_valid_reshaped[:, 4] = y_valid  # 将 y_valid 填充到最后一列
    y_valid_inverse = scaler.inverse_transform(y_valid_reshaped)
    y_valid_final = y_valid_inverse[:, 4]
    #y_valid = scaler.inverse_transform([y_valid])
    R_pre2 = model.predict(x_train)
    R_pre2 = R_pre2.flatten()
    R_pre2_reshaped = np.zeros((len(R_pre2), 5))  # 5 是原始特征的数量
    R_pre2_reshaped[:, 4] = R_pre2  # 将 y_valid 填充到最后一列
    R_pre2_inverse = scaler.inverse_transform(R_pre2_reshaped)
    R_pre2_final = R_pre2_inverse[:, 4]
    #R_pre2 = scaler.inverse_transform(R_pre2)
    y_train_reshaped = np.zeros((len(y_train), 5))  # 5 是原始特征的数量
    y_train_reshaped[:, 4] = y_train  # 将 y_valid 填充到最后一列
    y_train_inverse = scaler.inverse_transform(y_train_reshaped)
    y_train_final = y_train_inverse[:, 4]
    #y_train = scaler.inverse_transform([y_train])
    y_valid2 = np.transpose(y_valid_final)
    Rpre = np.column_stack((R_pre1_final,y_valid2)) 
    np.savetxt('/share/home/kong/github/CUG-hydro/HMs-zjw/test/'+ files[i], Rpre, delimiter=',')
    y_train2 = np.transpose(y_train_final)
    Rpre3 = np.column_stack((R_pre2_final,y_train2)) 
    np.savetxt('/share/home/kong/github/CUG-hydro/HMs-zjw/train/'+ files[i], Rpre3, delimiter=',')