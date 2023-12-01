import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from model import Model
from xin_feeder_baidu import Feeder
from datetime import datetime
import random
import itertools
import math
import pandas as pd

CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

max_x = 1.
max_y = 1.
history_frames = 6  # 3 second * 2 frame/second
future_frames = 6  # 3 second * 2 frame/second

batch_size_train = 64
batch_size_val = 32
batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0'
work_dir = './trained_models'
log_file = os.path.join(work_dir, 'log_test.txt')
test_result_file = 'prediction_result.txt'

criterion = torch.nn.SmoothL1Loss()

if not os.path.exists(work_dir):
    os.makedirs(work_dir)


def my_print(pra_content):
    with open(log_file, 'a') as writer:
        print(pra_content)
        writer.write(pra_content + '\n')


def display_result(pra_results, pra_pref='Train_epoch'):
    all_overall_sum_list, all_overall_num_list = pra_results
    overall_sum_time = np.sum(all_overall_sum_list ** 0.5, axis=0)
    overall_num_time = np.sum(all_overall_num_list, axis=0)
    overall_loss_time = (overall_sum_time / overall_num_time)
    overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(
        ['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
    my_print(overall_log)
    return overall_loss_time


def rmse_point(origin, predict):
    """
    :param origin: 原坐标
    :param predict: 预测坐标
    """
    xy_sum = 0.0
    for k in range(origin.shape[0]):
        xy_sum += (origin[k][0] - predict[k][0]) * (origin[k][0] - predict[k][0]) +\
                  (origin[k][1] - predict[k][1]) * (origin[k][1] - predict[k][1])

    xy_sum /= origin.shape[0]
    return math.sqrt(xy_sum)


if __name__ == '__main__':
    look_back = 10
    predict_data = np.array(pd.read_csv("./prediction_result.txt", sep=' '), dtype=np.float64)
    origin_data = np.array(pd.read_csv("./data/frame.txt", sep=' '), dtype=np.float64)
    dict_x = {}
    for i in range(origin_data.shape[0]):
        dict_x[str(origin_data[i][0]) + str(origin_data[i][1])] = [origin_data[i][8], origin_data[i][9]]

    predict_1s_rmse = rmse_point(d1s, p1s)
    predict_3s_rmse = rmse_point(d3s, p3s)
    predict_5s_rmse = rmse_point(d5s, p5s)

    print('predict_1s_rmse =', predict_1s_rmse, '\n')
    print('predict_3s_rmse =', predict_3s_rmse, '\n')
    print('predict_5s_rmse =', predict_5s_rmse, '\n')
