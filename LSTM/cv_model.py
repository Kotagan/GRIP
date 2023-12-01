import numpy as np
import pandas as pd
import math
import rmse


def cv_model(model, predict):
    """
    :param model: 原坐标
    :param predict: 预测坐标
    """

    xy_sum = 0.0
    for k in range(model.shape[0]):
        xy_sum += (model[k][0] - predict[k][0]) * (model[k][0] - predict[k][0]) + (model[k][1] - predict[k][1]) * (model[k][1] - predict[k][1])

    xy_sum /= model.shape[0]
    return math.sqrt(xy_sum)


if __name__ == '__main__':
    look_back = 10
    f = np.array(pd.read_csv("./test_x.csv"), dtype=np.float64)
    ff = np.array(pd.read_csv("./ktp.csv"), dtype=np.float64)
    dict_x = {}
    for i in range(ff.shape[0]):
        dict_x[ff[i][0]] = [ff[i][8], ff[i][9]]
    predict = f.reshape(-1, 5, 8)
    predict = predict.take([4], 1)
    predict = predict.reshape(-1, 8)

    result, d1s, d3s, d5s = np.zeros(shape=(predict.shape[0], 7)), np.zeros(shape=(predict.shape[0], 2)), \
        np.zeros(shape=(predict.shape[0], 2)), np.zeros(shape=(predict.shape[0], 2))
    for i in range(len(predict)):
        k = []
        v_x = math.cos(predict[i][6]) * predict[i][7]
        v_y = math.sin(predict[i][6]) * predict[i][7]
        result[i][0] = predict[i][2]
        # 预测1s
        result[i][1] = predict[i][4] + v_x
        result[i][2] = predict[i][5] + v_y

        # 预测3s
        result[i][3] = predict[i][4] + 3 * v_x
        result[i][4] = predict[i][5] + 3 * v_y
        # 预测5s
        result[i][5] = predict[i][4] + 5 * v_x
        result[i][6] = predict[i][5] + 5 * v_y

        # 查询原位置
        if int(result[i][0]) == 0:
            print("Fuck")
        d1s[i] = dict_x[int(result[i][0]) + look_back * 1]
        d3s[i] = dict_x[int(result[i][0]) + look_back * 3]
        d5s[i] = dict_x[int(result[i][0]) + look_back * 5]

    output = pd.DataFrame(result)
    output.to_csv('cv_predict.csv')

    p1s = result.take([1, 2], 1)
    p3s = result.take([3, 4], 1)
    p5s = result.take([5, 6], 1)

    predict_1s_rmse = rmse.rmse_point(d1s, p1s)
    predict_3s_rmse = rmse.rmse_point(d3s, p3s)
    predict_5s_rmse = rmse.rmse_point(d5s, p5s)

    print('predict_1s_rmse =', predict_1s_rmse, '\n')
    print('predict_3s_rmse =', predict_3s_rmse, '\n')
    print('predict_5s_rmse =', predict_5s_rmse, '\n')







