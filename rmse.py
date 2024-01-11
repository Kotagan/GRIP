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
import matplotlib.pyplot as plt
import re

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
dev = 'cuda:0'

history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second

prediction_data_root = ''
test_origin_data = 'data/frame1002-1006.txt'


if __name__ == '__main__':
    prediction_data_file_path = sorted(glob.glob(os.path.join(prediction_data_root, 'prediction_result.txt')))
    print('Prepare Prediction Data.')
    time_label = [0.5, 1, 1.5, 2, 2.5, 3]
    rmse_result_list = []
    for file_path in prediction_data_file_path:
        # re.findall(r'(\w+)(\d+)(\w+)', file_path)
        file_path = 'prediction_result.txt'
        predict_data = np.array(pd.read_csv(file_path, sep=' ', header=None), dtype=np.float64)

        test_total_data = np.array(pd.read_csv(test_origin_data, sep=' ', header=None), dtype=np.float64)
        origin_data_set = {}
        for row in test_total_data:
            origin_data_set[str(list[int(row[frame_id]), int(row[object_id])])] = row

        rmse_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        num_count = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        id_map = {}
        for row in predict_data:
            if not row[frame_id] in id_map:
                for j in range(6):
                    id_map[row[frame_id] + j] = j
            if not str(list[int(row[frame_id]), int(row[object_id])]) in origin_data_set:
                continue
            origin_data = origin_data_set[str(list[int(row[frame_id]), int(row[object_id])])]
            rmse_result[id_map[row[frame_id]]] += ((origin_data[position_x] - row[position_x])**2 +
                                                   (origin_data[position_y] - row[position_y])**2)
            num_count[id_map[row[frame_id]]] += 1

        for j in range(future_frames):
            rmse_result[j] = math.sqrt(rmse_result[j]/num_count[j])
            print("grip_rmse_" + str(j + 1) + "_frame = " + str(rmse_result[j]))

        rmse_result_list.append(rmse_result)

    # l1 = plt.plot(rmse_result_list[0], time_label, 'r+-', label='0')
    # l2 = plt.plot(rmse_result_list[1], time_label, 'g+-', label='5')
    # l3 = plt.plot(rmse_result_list[2], time_label, 'b^-', label='10')
    # l4 = plt.plot(rmse_result_list[3], time_label, 'b--', label='15')
    # l5 = plt.plot(rmse_result_list[4], time_label, 'r--', label='20')
    # plt.plot(rmse_result_list[0][1], time_label)
    # plt.plot(rmse_result_list[0][0], rmse_result_list[0][1], 'r+-',
    #          rmse_result_list[1][0], rmse_result_list[1][1], 'g+-',
    #          rmse_result_list[2][0], rmse_result_list[2][1], 'b^-',
    #          rmse_result_list[3][0], rmse_result_list[3][1], 'b--',
    #          rmse_result_list[4][0], rmse_result_list[4][1], 'r--'
    #          )
    # plt.title('The Rmse in each Epoch')
    # plt.xlabel('time step')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.show()



