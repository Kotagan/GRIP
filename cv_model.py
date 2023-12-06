import numpy as np
import pandas as pd
import math
import rmse

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

if __name__ == '__main__':
    cross_data_list = np.array(pd.read_csv("./frame.txt", sep=' '), dtype=np.float64)
    all_data_list = np.array(pd.read_csv("./frame_total.txt", sep=' '), dtype=np.float64)
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
            real_data = dict_x[str(list[predict_frame, int(row[object_id])])]
            rmse_result[j] += ((row[position_x] - real_data[position_x]) ** 2 +
                               (row[position_y] - real_data[position_y]) ** 2)
            rmse_count_num += 1

    for j in range(6):
        rmse_result[j] = math.sqrt(rmse_result[j] / rmse_count_num[j])
        print("cv_model_rmse_" + str(j+1) + "_frame = " + str(rmse_result[j]))
