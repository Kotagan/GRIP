import pandas as pd
import numpy as np
import math
look_back = 12


def create_dateset(trajectory):
    # [frame_id, object_id, object_type, position_x, position_y, position_z,
    # object_length, object_width, object_height, heading]

    data_x, data_y = [], []
    for i in range(len(trajectory)):
        if trajectory[i][2] != trajectory[i + 9 * look_back][2]:
            continue
        if trajectory[i][1] != trajectory[i + 9 * look_back][1] - 9 * look_back:
            continue

        for j in range(0, 5):
            data_x.append(trajectory[i + j * look_back])
        for j in range(5, 10):
            data_y.append(trajectory[i + j * look_back])

    data_x = np.array(data_x, dtype='float64')
    data_y = np.array(data_y, dtype='float64')
    data_x = data_x.reshape(-1, 35)
    data_y = data_y.reshape(-1, 35)
    data_x = np.hstack((data_x, data_y))
    train_x = data_x[:, :35]
    train_y = data_x[:, 35:]
    train_x = train_x.reshape(-1, 7)
    train_y = train_y.reshape(-1, 7)

    save = pd.DataFrame(train_x)
    save.to_csv('train_x.csv')

    return train_x, train_y


def normalize(data):
    """
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    """
    normalize_data = np.arange(2 * data.shape[1], dtype='float64')
    normalize_data = normalize_data.reshape(data.shape[1], 2)

    for i in range(0, data.shape[1]):
        if i == 0 or i == 1 or i == 2:
            continue
        data_list = data[:, i]

        normalize_data[i, 0] = np.max(data_list)
        normalize_data[i, 1] = np.min(data_list)
    return normalize_data


def normalize_with_data(data, normalize_data):
    """
    归一化 适用于单维和多维
    返回归一化后的数据和最大最小值
    """
    for i in range(0, data.shape[1]):
        list_max = normalize_data[i, 1]
        list_min = normalize_data[i, 0]
        delta = list_max - list_min
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - list_min) / delta
    return data


def denormalize(data, normalize_data):
    # return data * (max_val - min_val) + min_val
    for i in range(0, data.shape[1]):
        normalize_min = normalize_data[i, 0]
        normalize_max = normalize_data[i, 1]
        for j in range(0, data.shape[0]):
            data[j, i] = data[j, i] * (normalize_max - normalize_min) + normalize_min
    return data


def load_data():
    # 从文件中取出训练数据
    data_csv = pd.read_csv("./lstm.csv", header=True)
    trajectory = np.array(data_csv, dtype=np.float64)
    trajectory = trajectory.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)

    # 进行归一化
    normalize_data = normalize(trajectory)
    np.save("./normalize.npy", normalize_data)

    data_x, data_y = create_dateset(trajectory)
    data_x = normalize_with_data(data_x, normalize_data)
    data_y = normalize_with_data(data_y, normalize_data)

    data_x = data_x.take([4, 5], 1)
    data_y = data_y.take([4, 5], 1)
    train_x = data_x.reshape(-1, 5, 2)
    train_y = data_y.reshape(-1, 10)

    return train_x, train_y


def load_test_data(normalize_data):
    # 取前data_size个数据作为预测对象
    data_x = pd.read_csv("./test_x.csv")
    data_x = np.array(data_x, dtype=np.float64)
    data_x = data_x.take([1, 2, 3, 4, 5, 6, 7], 1)
    data_x = normalize_with_data(data_x, normalize_data)

    test_x = data_x.take([3, 4], 1)
    test_x = test_x.reshape(-1, 5, 2)

    return test_x, data_x

