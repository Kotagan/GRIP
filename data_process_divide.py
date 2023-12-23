import numpy as np
import glob
import os
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

# Please change this to your location
data_root = './data/prediction_train/'
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


def generate_origin_data(file_path):
    """
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feature: (N, C, T, V)
            N is the number of training data
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for fewer objects.
    """

    all_data_list = np.array(pd.read_table(file_path, sep=' ', header=None, index_col=False))
    all_data_list = all_data_list.take([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1)
    data_list = []
    frame_id_set = {}
    num = 0
    for row in all_data_list:
        if not (int(row[0]) - 1 in frame_id_set
                or int(row[0]) in frame_id_set
                or len(data_list) == 0):
            if data_list[len(data_list) - 1][0] - data_list[0][0] < 12:
                data_list = []
            else:
                pd.DataFrame(data_list).to_csv('./data/prediction_train/' + str(num) + '.txt', sep=' ', index=False, header=False)
                data_list = []
                num += 1
        data_list.append(row)
        frame_id_set[int(row[0])] = 0
    pd.DataFrame(data_list).to_csv('./data/prediction_train/' + str(num) + '.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    data_file_path = os.path.join(data_root, 'frame_02-10_test.txt')
    print('Dividing Data.')
    generate_origin_data(data_file_path)
