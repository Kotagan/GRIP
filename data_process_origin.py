import math

import numpy as np
import glob
import os
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

# Please change this to your location
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


def compare_which_is_close(former_time, later_time):
    frame_time = round(former_time / 500)
    return (math.fabs(frame_time - round(former_time / 500) * 500) <
            math.fabs(frame_time - round(later_time / 500) * 500))


def get_origin_data_list(pra_file_path):
    """
    Read raw data from files and return a dictionary:
        {frame_id:
            {object_id:
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length,
                object_width, object_height, heading]
            }
        }
    """

    map_id = {}  # record object id
    pair_list = {}  # record same frame + id
    id_count = 1

    for file_path in pra_file_path:
        content = np.array(
            pd.read_csv(file_path, header=None).to_numpy()[:, [1, 0, 2, 20, 21, 22, 52, 54, 56, 50, 44]])
        for row in content:
            # change object_id(str) to object_id(int)
            if not (row[object_id] in map_id):
                map_id[row[object_id]] = id_count
                id_count += 1
            row[object_id] = map_id[row[object_id]]

            data_frame_id = int(round(row[frame_id] / 500, 0))

            # if pair(frame_id, object_id) exist, continue
            if str(list[int(data_frame_id), int(row[object_id])]) in pair_list and compare_which_is_close(pair_list[list(data_frame_id, row[object_id])][frame_id], row[frame_id]):
                continue
            pair_list[str([data_frame_id, row[object_id]])] = row

    data_list = []
    for index, key in enumerate(pair_list):
        row = pair_list[key]
        # millisecond → frame(2frame/second)
        row[frame_id] = int(round(int(row[frame_id]) / 500))
        # if row[position_x] <= -9724.29 or row[position_y] <= -79217.14:
        #     continue

        row[object_width] = 1.7
        row[object_length] = 4
        row[object_height] = 1.5

        # data process
        if row[object_type] == 1:
            row[object_type] = 2
        elif row[object_type] == 2:
            row[object_type] = 3
        elif row[object_type] == 4:
            row[object_type] = 4
        else:
            print('error')

        data_list.append(row)

    pd.DataFrame(data_list).to_csv('./data/prediction_train/frame.txt', sep=' ',
                                    index=False, header=False)
    return


if __name__ == '__main__':
    origin_data_file_path = sorted(glob.glob(os.path.join(data_root, '*.csv')))
    print('Generating Origin Data.')
    get_origin_data_list(origin_data_file_path)
