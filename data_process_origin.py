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
            pd.read_csv(file_path, header=1).to_numpy()[:, [10, 15, 0, 1, 0, 1, 1, 1, 1, 19, 18]])
        for row in content:
            # change object_id(str) to object_id(int)
            if not (row[object_id] in map_id):
                map_id[row[object_id]] = id_count
                id_count += 1
            row[object_id] = map_id[row[object_id]]
            if row[frame_id] % 5 != 0:
                continue
            row[frame_id] = int(row[frame_id] / 5)
            # if pair(frame_id, object_id) exist, continue
            pair_list[str([row[frame_id], row[object_id]])] = row

    data_list = []
    for index, key in enumerate(pair_list):
        row = pair_list[key]
        # if row[position_x] <= -9724.29 or row[position_y] <= -79217.14:
        #     continue

        row[object_width] = 1.7
        row[object_length] = 4
        row[object_height] = 1.5
        row[object_type] = 1
        row[position_z] = 0
        # data process

        data_list.append(row)

    pd.DataFrame(data_list).to_csv('./data/prediction_train/frame.txt', sep=' ',
                                   index=False, header=False)
    return


if __name__ == '__main__':
    origin_data_file_path = sorted(glob.glob(os.path.join(data_root, '*.csv')))
    print('Generating Origin Data.')
    get_origin_data_list(origin_data_file_path)
