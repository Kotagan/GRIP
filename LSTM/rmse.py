import numpy as np
import pandas as pd
import math


def rmse_point(model, predict):
    """
    :param model: 原坐标
    :param predict: 预测坐标
    """

    xy_sum = 0.0
    for k in range(model.shape[0]):
        xy_sum += (model[k][0] - predict[k][0]) * (model[k][0] - predict[k][0]) +\
                  (model[k][1] - predict[k][1]) * (model[k][1] - predict[k][1])

    xy_sum /= model.shape[0]
    return math.sqrt(xy_sum)


if __name__ == '__main__':
    look_back = 10
    fy = np.array(pd.read_csv("./predict_y.csv"), dtype=np.float64)
    ff = np.array(pd.read_csv("./ktp.csv"), dtype=np.float64)
    dict_x = {}
    for i in range(ff.shape[0]):
        dict_x[ff[i][0]] = [ff[i][8], ff[i][9]]

    p1s = fy.take([2, 3], 1)
    p3s = fy.take([6, 7], 1)
    p5s = fy.take([10, 11], 1)

    d1s, d3s, d5s = np.zeros(shape=(fy.shape[0], 2)), np.zeros(shape=(fy.shape[0], 2)), np.zeros(shape=(fy.shape[0], 2))
    for i in range(fy.shape[0]):
        d1s[i] = dict_x[int(fy[i][1]) + look_back * 1]
        d3s[i] = dict_x[int(fy[i][1]) + look_back * 3]
        d5s[i] = dict_x[int(fy[i][1]) + look_back * 5]

    predict_1s_rmse = rmse_point(d1s, p1s)
    predict_3s_rmse = rmse_point(d3s, p3s)
    predict_5s_rmse = rmse_point(d5s, p5s)

    print('predict_1s_rmse =', predict_1s_rmse, '\n')
    print('predict_3s_rmse =', predict_3s_rmse, '\n')
    print('predict_5s_rmse =', predict_5s_rmse, '\n')

