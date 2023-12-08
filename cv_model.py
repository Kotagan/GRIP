import numpy as np
import pandas as pd
import math
import glob
import os

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
speed = 10

if __name__ == '__main__':
    cross_data_list = np.array(pd.read_csv("./frame_cross_data.txt", sep=' '), dtype=np.float64)
    all_data_list = np.array(pd.read_csv("./frame_total_data.txt", sep=' '), dtype=np.float64)
    dict_x = {}
    for row in all_data_list:
        dict_x[str(list[int(row[frame_id]), int(row[object_id])])] = row

    rmse_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rmse_count_num = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for row in cross_data_list:
        for j in range(6):
            predict_frame = int(row[frame_id] + j + 1)
            if not str(list[predict_frame, int(row[object_id])]) in dict_x:
                continue
            predict_x_position = row[position_x] + 0.5 * row[speed] * math.cos(row[heading]) * (j + 1)
            predict_y_position = row[position_y] + 0.5 * row[speed] * math.sin(row[heading]) * (j + 1)
            real_x_position = dict_x[str(list[predict_frame, int(row[object_id])])][position_x]
            real_y_position = dict_x[str(list[predict_frame, int(row[object_id])])][position_y]
            rmse_result[j] += (real_x_position - predict_x_position) ** 2 + (real_y_position - predict_y_position) ** 2
            rmse_count_num[j] += 1

    for j in range(6):
        rmse_result[j] = math.sqrt(rmse_result[j] / rmse_count_num[j])
        print("cv_model_rmse_" + str(j+1) + "_frame = " + str(rmse_result[j]))
