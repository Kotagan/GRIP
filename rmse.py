import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from model import Model
from xin_feeder_baidu import Feeder
from datetime import datetime
import random
import itertools
import math
import pandas as pd
import glob

data_root = './data/'
frame_id = 0
object_id = 1
object_type = 2
position_x = 3
position_y = 4
position_z = 5
object_length = 6
object_width = 7
object_height = 8
heading = 9
CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

max_x = 1.
max_y = 1.
history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second

test_result_file = 'prediction_result.txt'
test_origin_data = 'data/frame_02-10_all.txt'

if __name__ == '__main__':
    predict_data = np.array(pd.read_csv(test_result_file, sep=' ', header=None), dtype=np.float64)

    test_total_data = np.array(pd.read_csv(test_origin_data, sep=' ', header=None), dtype=np.float64)
    origin_data_set = {}
    for row in test_total_data:
        origin_data_set[str(list[int(row[frame_id]), row[object_id]])] = row

    predict_data = predict_data.reshape(-1, 6, 5)
    rmse_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    countn = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(predict_data.shape[0]):
        for k in range(future_frames):
            if not str(list[int(predict_data[i][k][frame_id]), predict_data[i][k][object_id]]) in origin_data_set:
                continue
            countn[k] += 1
            origin_data = origin_data_set[str(list[int(predict_data[i][k][frame_id]), predict_data[i][k][object_id]])]
            # print(((float(origin_data[position_x]) - (float(predict_data[i][k][position_x]))) * ((float(origin_data[position_y])) - (float(predict_data[i][k][position_y])))) ** 2)
            rmse_result[k] += ((float(origin_data[position_x]) - (float(predict_data[i][k][position_x]))) * ((float(origin_data[position_y])) - (float(predict_data[i][k][position_y])))) ** 2

    for j in range(future_frames):
        rmse_result[j] = math.sqrt(rmse_result[j]/countn[j])
    print(rmse_result)
