import math

import numpy as np
import glob
import os
import pandas as pd
from pyproj import CRS
from pyproj import Transformer

# Please change this to your location
data_root = './data/'
time_frame = 1
object_id = 0
object_type = 2
position_x = 20
position_y = 21
position_z = 22
object_length = 52
object_width = 54
object_height = 56
heading = 50
speed = 44


def get_origin_data_list(pra_file_path):
    map_id = {}  # record object id

    crs_wgs84 = CRS.from_epsg(4326)
    crs_jgd2011 = CRS.from_epsg(6675)
    converter = Transformer.from_crs(crs_wgs84, crs_jgd2011)

    for file_path in pra_file_path:
        data_map = {}
        data_list = []
        content = np.array(pd.read_csv(file_path, header=None).to_numpy())
        for row in content:
            row[position_x], row[position_y] = converter.transform(row[position_x] / 10000000,
                                                                   row[position_y] / 10000000)
            row[position_x], row[position_y] = row[position_y], row[position_x]
            if not (row[object_id] in map_id):
                map_id[row[object_id]] = row[object_id]
                data_map[row[object_id]] = row
                data_list.append(row)
                continue
            former_data = data_map[row[object_id]]
            data_map[row[object_id]] = row
            if str(row) == str(former_data):
                continue

            if former_data[position_x] == former_data[position_x]:
                row[heading] = math.pi / 2
                if former_data[position_y] > row[position_y]:
                    row[heading] = -row[heading]
            else:
                row[heading] = math.atan((former_data[position_y] - row[position_y]) /
                                         (former_data[position_x] - former_data[position_x]))
            if row[time_frame] - former_data[time_frame] == 0:
                row[speed] = 0
            else:
                row[speed] = (math.sqrt((former_data[position_y] - row[position_y])**2 +
                                        (former_data[position_x] - former_data[position_x])**2) /
                              ((row[time_frame] - former_data[time_frame]) / 1000))
            data_map[map_id[row[object_id]]] = row
            data_list.append(row)

        pd.DataFrame(data_list).to_csv(file_path, sep=',', index=False, header=False)

    return


if __name__ == '__main__':
    origin_data_file_path = sorted(glob.glob(os.path.join(data_root, '*.csv')))
    print('Generating Origin Data.')
    get_origin_data_list(origin_data_file_path)
