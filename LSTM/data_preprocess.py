import csv

import numpy as np
import pandas as pd
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

def compare_which_is_close(former_time, later_time):
    frame_time = round(former_time / 500)
    return (math.fabs(frame_time - round(former_time / 500) * 500) <
            math.fabs(frame_time - round(later_time / 500) * 500))
# ktpの前処理
if __name__ == '__main__':
    # [frame_id, object_id, object_type, position_x, position_y, position_z, object_length,
    # object_width, object_height, heading]
    content = np.array(pd.read_csv("./../data/object_info_20231005_sorted.csv", header=None).to_numpy()[:,
                       [1, 0, 2, 20, 21, 22, 52, 54, 56, 50]])

    map_id = {}  # record object id
    map_list = {}  # record same data
    pair_list = {}  # record same frame + id
    id_count = 1
    for row in content:
        if str(row) in map_list:
            continue
        map_list[str(row)] = 0

        # change object_id(str) to object_id(int)
        if not (row[object_id] in map_id):
            map_id[row[object_id]] = id_count
            id_count += 1
        row[object_id] = map_id[row[object_id]]

        data_frame_id = round(row[frame_id] / 500, 0)

        # if pair(frame_id, object_id) exist, continue
        if str(list[data_frame_id, row[object_id]]) in pair_list and compare_which_is_close(pair_list[list(data_frame_id, row[object_id])][frame_id], row[frame_id]):
            continue
        pair_list[str(list[row[frame_id], row[object_id]])] = row
    save = pd.DataFrame(content)
    save.to_csv('lstm.csv')
