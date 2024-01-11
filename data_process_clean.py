import math
import re

import numpy as np
import glob
import os
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

# Please change this to your location
data_root = './data'

time_frame = 16
object_id = 15
object_type = 0
position_x = 1
position_y = 0
position_z = 1
object_length = 1
object_width = 1
object_height = 1
heading = 4
speed = 2


def get_origin_data_list(pra_file_path):
    for file_path in pra_file_path:
        map_id = {}  # record object id
        data_map = {}
        data_list = []
        content = np.array(pd.read_csv(file_path, dtype={'smooth_x': np.float64,
                                                         'smooth_y': np.float64,
                                                         'smooth_v': np.float64,
                                                         'diff_v': np.float64,
                                                         'smooth_h': np.float64,
                                                         'diff_h': np.float64,
                                                         'smooth_a': np.float64,
                                                         'diff_a': np.float64,
                                                         'timestep': np.float64,
                                                         'ori': np.float64,
                                                         'id': object,
                                                         }).to_numpy())

        for row in content:
            row[position_x], row[position_y] = row[position_y], row[position_x]
            row[time_frame] = round(row[time_frame] / 100)
            # print(row[time_frame])
            if not (row[object_id] in map_id):
                map_id[row[object_id]] = row[object_id]
                data_map[row[object_id]] = row
                data_list.append(row)
                continue

            former_data = data_map[row[object_id]]
            if former_data[time_frame] == row[time_frame] - 1:
                data_map[row[object_id]] = row
                data_list.append(row)
                continue

            mid_data = row.copy()

            for k in range(len(mid_data)):
                if not isinstance(mid_data[k], str):
                    mid_data[k] = (former_data[k] + row[k]) / 2

            data_list.append(mid_data)
            data_list.append(row)
            data_map[row[object_id]] = row

        file_path = file_path[:-4] + "_new.csv"
        pd.DataFrame(data_list).to_csv(file_path, sep=',', index=False, header=True)

    return


if __name__ == '__main__':
    origin_data_file_path = sorted(glob.glob(os.path.join(data_root, '*.csv')))
    print('Generating Origin Data.')
    get_origin_data_list(origin_data_file_path)
