import numpy as np
import glob
import os
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

# Please change this to your location
data_train_root = './data/prediction_train/'
data_test_root = './data/prediction_test/'
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


def generate_origin_data(file_path, data_root):
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
        if not (len(data_list) == 0 or int(row[0]) - 1 in frame_id_set or int(row[0]) in frame_id_set):
            # reset_frame_id = data_list[0][frame_id]
            # for data in data_list:
            #     data[frame_id] -= reset_frame_id
            # data_list.sort(key=lambda x: x[frame_id])
            # while len(data_list) != 0 and data_list[0][frame_id] % 12 != 1:
            #     data_list = data_list[1:].copy()
            # while len(data_list) != 0 and data_list[len(data_list)-1][frame_id] % 12 != 0:
            #     data_list = data_list[:-1].copy()

            if len(data_list) != 0:
                pd.DataFrame(data_list).to_csv(data_root + str(num) + '.txt', sep=' ', index=False, header=False)
                data_list = []
                frame_id_set = {}
                num += 1
        data_list.append(row)
        frame_id_set[int(row[0])] = 0
    pd.DataFrame(data_list).to_csv(data_root + str(num) + '.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    data_file_path = os.path.join(data_train_root, 'frame.txt')
    print('Dividing Train Data.')
    generate_origin_data(data_file_path, data_train_root)
    data_file_path = os.path.join(data_test_root, 'frame.txt')
    print('Dividing Test Data.')
    generate_origin_data(data_file_path, data_test_root)
