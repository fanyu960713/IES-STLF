import pandas as pd
import numpy as np
import torch
def getDataforTrain():
    path = 'data.csv'
    total = torch.tensor(np.array(pd.read_csv(path)))
    _x_data = total[:162 * 24, :3].unsqueeze(1)
    _x_data = _x_data.reshape(162*24 , 1, 3)
    _y_data = total[:162 * 24, 3:6].unsqueeze(1)
    _y_data = _y_data.reshape(162*24 , 1, 3)
    return _x_data , _y_data

def getDataforTest():
    path = 'data.csv'
    total = torch.tensor(np.array(pd.read_csv(path)))
    _x_data = total[162 * 24:180 * 24, :3].unsqueeze(1)
    _x_data = _x_data.reshape(18 * 24, 1, 3)
    _y_data = total[162 * 24:180 * 24, 3:6].unsqueeze(1)
    _y_data = _y_data.reshape(18 * 24, 1, 3)
    return _x_data , _y_data