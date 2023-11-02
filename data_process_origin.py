import numpy as np
import glob
import os
import pandas as pd

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


def get_origin_data_list(pra_file_path):
    '''
    Read raw data from files and return a dictionary:
        {frame_id:
            {object_id:
                # 10 features
                [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
            }
        }
    '''
    frame_list = np.empty(shape=[0, 10])
    content = np.array(pd.read_csv(pra_file_path, header=None).to_numpy()[:, [1, 0, 4, 20, 21, 22, 52, 54, 56, 50]])
    map_id = {}  # record object id
    map_list = {}  # record same data
    pair_list = {}  # record same frame + id
    time_dict = {}
    id_count = 1
    first = 0
    last = 0
    temp_first = 0
    temp_last = 0
    for row in content:
        # millisecond → frame(2frame/second)
        row[0] = int(round(row[0] / 500, 0))
        time_dict[row[0]] = {}

        # delete same data
        if str(row) in map_list:
            continue
        map_list[str(row)] = 0

        if not (row[1] in map_id):
            map_id[row[1]] = id_count
            id_count += 1
        row[1] = map_id[row[1]]

        if str(list[row[0], row[1]]) in pair_list:
            continue
        pair_list[str(list[row[0], row[1]])] = 0

        temp = row.reshape(-1, 10)
        frame_list = np.concatenate([frame_list, temp], 0)

    for index in range(len(frame_list)):
        row = frame_list[index]
        #  find longest sequence
        if not (row[0] - 1 in time_dict):
            if temp_last - temp_first > first - last:
                first = temp_first
                last = temp_last
            temp_first = index
            temp_last = index
        else:
            temp_last = index
    return frame_list[first:last]


def generate_origin_data(file_path_list, ):
    """
    Read data from $pra_file_path, and split data into clips with $total_frames length.
    Return: feature and adjacency_matrix
        feature: (N, C, T, V)
            N is the number of training data
            C is the dimension of features, 10raw_feature + 1mark(valid data or not)
            T is the temporal length of the data. history_frames + future_frames
            V is the maximum number of objects. zero-padding for less objects.
    """

    all_data_list = np.array([])
    for file_path in file_path_list:
        now_list = get_origin_data_list(file_path)
        all_data_list = np.array(now_list)

    all_data_list_length = int(len(all_data_list) * 0.8)
    train_list = all_data_list[:all_data_list_length]
    test_list = all_data_list[all_data_list_length:]
    first_frame_id=train_list[0][0]
    for row in train_list:
        row[0] -= first_frame_id
    first_frame_id = test_list[0][0]
    for row in test_list:
        row[0] -= first_frame_id

    pd.DataFrame(train_list).to_csv('./data/prediction_train/frame.txt', sep=' ', index=False, header=False)
    pd.DataFrame(test_list).to_csv('./data/prediction_test/frame.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    origin_data_file_path_list = sorted(glob.glob(os.path.join(data_root, '*.csv')))
    print('Generating Origin Data.')
    generate_origin_data(origin_data_file_path_list)
