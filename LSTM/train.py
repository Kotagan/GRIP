import data as dt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Embedding, LSTM, Dropout, Activation
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


def train_model(train_x, train_y, epochs, batch_size, dropout):
    """
    :param train_x:
    :param train_y: LSTM训练所需的训练集　訓練データ
    :return: 训练得到的模型　訓練モデル
    """

    model = Sequential()
    model.add(LSTM(
        256,
        input_shape=(train_x.shape[1], train_x.shape[2]),
        return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(
        256,
        return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(train_y.shape[1], activation='relu'))  # 10个输出的全连接层


    model.compile(loss='mse', optimizer='adam', metrics=['acc'])
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, \
                        validation_split=0.2, verbose=1)
    model.summary()

    # plot train and validation loss
    plt.figure(figsize=(8, 8), dpi=200)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_acc'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'train_acc', 'validation_loss', 'validation_acc'], loc='upper right')
    plt.show()
    return model


if __name__ == '__main__':
    # 環境設定　环境设定
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    train_x, train_y = dt.load_data()

    epochs = 200
    batch_size = 8
    dropout = 0.2
    model = train_model(train_x, train_y, epochs, batch_size, dropout)
    model_name = "./model_new/eps_" + str(epochs) + "_bs_" + str(batch_size) + "_dp_" + str(dropout) + "(256).h5"
    model.save(model_name)
