import pandas as pd
import numpy as np
import tensorflow as tf
import data as dt
import os

if __name__ == '__main__':
    normalize_data = np.load("./normalize.npy")

    # predict
    test_x, data_x = dt.load_test_data(normalize_data)
    model = tf.keras.models.load_model("model_new/eps_200_bs_8_dp_0.2(256).h5")
    test_y = model.predict(test_x)
    test_y = test_y.reshape(-1, 2)
    test_x = test_x.reshape(-1, 2)

    # 反归一化
    test_x = dt.denormalize(test_x, normalize_data)
    test_y = dt.denormalize(test_y, normalize_data)
    test_y = test_y.reshape(-1, 10)
    test_x = test_x.reshape(-1, 10)
    # for i in range(len(test_y)):
    #     test_y[i][0] += -9762.0
    #     test_y[i][1] += -79132.0
    #     test_x[i][0] += -9762.0
    #     test_x[i][1] += -79132.0
    #     test_x[i][2] += -9762.0
    #     test_x[i][3] += -79132.0
    #     test_x[i][4] += -9762.0
    #     test_x[i][5] += -79132.0
    #     test_x[i][6] += -9762.0
    #     test_x[i][7] += -79132.0
    #     test_x[i][8] += -9762.0
    #     test_x[i][9] += -79132.0

    data_x = data_x.take([1], 1)
    data_x = data_x.reshape(-1, 5)
    data_x = data_x.take([4], 1)
    # test_x = np.hstack([data_x, test_x])
    test_y = np.hstack([data_x, test_y])

    save_x = pd.DataFrame(test_x, columns=[
                                           '-5s_x', '-5s_y',
                                           '-4s_x', '-4s_y',
                                           '-3s_x', '-3s_y',
                                           '-2s_x', '-5s_y',
                                           '-1s_x', '-1s_y'])

    save_y = pd.DataFrame(test_y, columns=['0s_id',
                                           '1s_x', '1s_y',
                                           '2s_x', '2s_y',
                                           '3s_x', '3s_y',
                                           '4s_x', '4s_y',
                                           '5s_x', '5s_y', ])
    save_x.to_csv('predict_x.csv')
    save_y.to_csv('predict_y.csv')

    wkt_string = "WKT\n"
    for i in range(test_x.shape[0]):
        wkt_string += "\"LINESTRING ("
        for j in range(0, 5):
            wkt_string += str(test_x[i][j * 2]) + " " + str(test_x[i][j * 2 + 1]) + ","
        for j in range(0, 4):
            wkt_string += str(test_y[i][j * 2 + 1]) + " " + str(test_y[i][j * 2 + 2]) + ","
        wkt_string += str(test_y[i][4 * 2 + 1]) + " " + str(test_y[i][4 * 2 + 2])
        wkt_string += ")\"\n"

    os.remove('out.wkt')
    file = open('out.wkt', 'w')
    file.write(wkt_string)
    file.close()


