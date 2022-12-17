##
## Created by tioeare on 12/04/22.
##
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler


class Lstm(object):
    """
    使用LSTM模型对数据进行训练

    Attributes:
        tran_stamp: 时间步长(即以过去多少天预测未来多少天中的过去的天数)
        pre_stamp: 预测天数(即预测未来多少天的数据)
        x_train: 训练数据
        y_train: 训练标号
        x_valid: 验证数据
        y_valid: 验证标号
        x_test: 测试数据
        y_test: 测试标号
    """

    def __init__(self):
        self.tran_stamp = 50
        self.pre_stamp = 5
        self.x_train, self.y_train = [], []
        self.x_valid, self.y_valid = [], []
        self.x_test, self.y_test = [], []

    def read_data(self, fileName=[]):
        """从文件读取数据

        Args:
            fileName (list): 读取文件名的列表

        Returns:
            data1 (numpy.array): 将汇率，原油，黄金和股票数据组合为训练数据
            label (numpy.array): 数据的标注集，表示为股票的经调整收市价
        """
        with open(fileName[0], encoding='utf-8') as f:
            stock_data = json.load(f)
        with open(fileName[1], encoding='utf-8') as f:
            Oil_data = json.load(f)
        with open(fileName[2], encoding='utf-8') as f:
            Gold_data = json.load(f)
        with open(fileName[3], encoding='utf-8') as f:
            CNY_data = json.load(f)
        with open(fileName[4], encoding='utf-8') as f:
            USD_data = json.load(f)
        # {[low], [high], [open], [close]}
        data = np.array([[stock_data['low'], Gold_data['low'], Oil_data['low'], CNY_data['low'],
                          USD_data['low']],
                         [stock_data['high'], Gold_data['high'], Oil_data['high'], CNY_data['high'],
                          USD_data['high']],
                         [stock_data['open'], Gold_data['open'], Oil_data['open'], CNY_data['open'],
                          USD_data['open']],
                         [stock_data['close'], Gold_data['close'], Oil_data['close'], CNY_data['close'],
                          USD_data['close']]])
        label = np.array(stock_data['adjclose'])
        stock_data = np.array([stock_data['low'], stock_data['high'], stock_data['open'], stock_data['close']])
        self.stock_data = stock_data.swapaxes(0, 1)
        data1 = data.swapaxes(0, 2)  # 调换维度，将时间顺序放在第一维，便于后续切片
        return data1, label

    def divide_data(self, data, label):
        """将数据切分为训练集，验证集和测试集

        Args:
            data (numpy.array): 训练数据
            label (numpy.array): 数据的标注集，表示为股票的经调整收市价
        """
        # 以7:1:1划分训练集，验证集和测试集
        n = int(data.shape[0] / 9)
        train_range = n * 7
        valid_range = n
        test_range = n
        train = data[:train_range - 1, :, :]
        valid = data[train_range - 1:train_range + valid_range - 1, :, :]
        test = data[train_range + valid_range - 1:train_range + valid_range + test_range - 1, :, :]
        # 归一化防止梯度爆炸
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_label = scaler.fit_transform(label.reshape(-1, 1))
        # 训练集切片
        scaled_data = scaler.fit_transform(train.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, data.shape[1], data.shape[2])
        for i in range(self.tran_stamp, len(train)):
            self.x_train.append(scaled_data[i - self.tran_stamp:i, :, :])
            self.y_train.append(scaled_label[i:i + self.pre_stamp])
        self.x_train, self.y_train = np.array(self.x_train).reshape(len(self.x_train), self.tran_stamp,
                                                                    -1), np.array(self.y_train).reshape(-1,
                                                                                                        self.pre_stamp)
        # 验证集切片
        scaled_data = scaler.fit_transform(valid.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, data.shape[1], data.shape[2])
        for i in range(self.tran_stamp, len(valid)):
            self.x_valid.append(scaled_data[i - self.tran_stamp:i, :, :])
            self.y_valid.append(scaled_label[i + train_range:i + train_range + self.pre_stamp])
        self.x_valid, self.y_valid = np.array(self.x_valid).reshape(len(self.x_valid), self.tran_stamp, -1), np.array(
            self.y_valid).reshape(-1, self.pre_stamp)
        # 测试集切片
        scaled_data = scaler.fit_transform(test.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, data.shape[1], data.shape[2])
        for i in range(self.tran_stamp, len(test)):
            self.x_test.append(scaled_data[i - self.tran_stamp:i, :, :])
            self.y_test.append(
                scaled_label[i + train_range + valid_range:i + train_range + valid_range + self.pre_stamp])
        self.x_test, self.y_test = np.array(self.x_test).reshape(len(self.x_valid), self.tran_stamp, -1), np.array(
            self.y_test).reshape(
            -1, self.pre_stamp)

    def lstm(self, label):
        """训练模型，包括训练和测试

        Args:
            label (numpy.array): 数据的标注集
        """
        # 如果设备上有nvidia的gpu则使用gpu训练
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        # 构建五层LSTM，前三层均为lstm，第四层防止过拟合，第五层全连接预测
        model = keras.Sequential()
        model.add(layers.LSTM(128, return_sequences=True, input_shape=(np.array(self.x_train).shape[1:])))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(self.pre_stamp))
        # 使用Adam做优化器，迭代最多70次
        model.compile(optimizer=keras.optimizers.Adam(), loss='mae', metrics=['accuracy'])
        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7,
                                                                    min_lr=0.0000000003)
        history = model.fit(np.array(self.x_train), np.array(self.y_train),
                            batch_size=128,
                            epochs=70,
                            validation_data=(self.x_valid, self.y_valid),
                            callbacks=[learning_rate_reduction])
        # 预测并将结果可视化
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_price = model.predict(self.x_test)
        model.evaluate(self.x_test)
        scaler.fit_transform(label.reshape(-1, 1))
        # 反归一化
        closing_price = scaler.inverse_transform(closing_price.reshape(-1, self.pre_stamp)[:, 0].reshape(1, -1))
        y_test = scaler.inverse_transform(np.array(self.y_test).reshape(-1, self.pre_stamp)[:, 0].reshape(1, -1))
        # 计算预测结果
        rms = np.sqrt(np.mean(np.power((y_test[0:1, self.pre_stamp:] - closing_price[0:1, self.pre_stamp:]), 2)))
        print('INFO: 均方根误差 ' + str(rms))
        # loss变化趋势可视化
        plt.figure(figsize=(27, 9))
        plt.subplot(1, 2, 1)
        plt.title('loss')
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='valid loss')
        plt.legend()
        # 预测效果可视化
        plt.subplot(1, 2, 2)
        plt.title('predict')
        plt.plot(closing_price.reshape(1, -1)[0], linewidth=3, alpha=0.8, label='predict_data')
        plt.plot(y_test[0], linewidth=1.2, label='true_data')
        plt.legend()
        # 保存预测数据
        np.savetxt('./Data/predict.csv', closing_price.reshape(1, -1)[0], delimiter=',')
        np.savetxt('./Data/true.csv', y_test[0], delimiter=',')

    def lstm_just_stock(self, label):
        """训练模型，包括训练和测试，只对股票数据进行训练

        Args:
            label (numpy.array): 数据的标注集
        """
        x_train, x_valid, x_test = [], [], []
        n = int(self.stock_data.shape[0] / 9)
        train_range = n * 7
        valid_range = n
        test_range = n
        train = self.stock_data[:train_range - 1, :]
        valid = self.stock_data[train_range - 1:train_range + valid_range - 1, :]
        test = self.stock_data[train_range + valid_range - 1:train_range + valid_range + test_range - 1, :]
        # 归一化防止梯度爆炸
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_label = scaler.fit_transform(label.reshape(-1, 1))
        # 训练集切片
        scaled_data = scaler.fit_transform(train.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, self.stock_data.shape[1])
        for i in range(self.tran_stamp, len(train)):
            x_train.append(scaled_data[i - self.tran_stamp:i, :])
        x_train = np.array(x_train).reshape(len(x_train), self.tran_stamp, -1)
        # 验证集切片
        scaled_data = scaler.fit_transform(valid.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, self.stock_data.shape[1])
        for i in range(self.tran_stamp, len(valid)):
            x_valid.append(scaled_data[i - self.tran_stamp:i, :])
        x_valid = np.array(x_valid).reshape(len(x_valid), self.tran_stamp, -1)
        # 测试集切片
        scaled_data = scaler.fit_transform(test.reshape(-1, 1))
        scaled_data = scaled_data.reshape(-1, self.stock_data.shape[1])
        for i in range(self.tran_stamp, len(test)):
            x_test.append(scaled_data[i - self.tran_stamp:i, :])
        x_test = np.array(x_test).reshape(len(x_valid), self.tran_stamp, -1)

        # 构建五层LSTM，前三层均为lstm，第四层防止过拟合，第五层全链接预测
        model = keras.Sequential()
        model.add(layers.LSTM(128, return_sequences=True, input_shape=(np.array(x_train).shape[1:])))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(self.pre_stamp))
        # 使用Adam做优化器，迭代最多70次
        model.compile(optimizer=keras.optimizers.Adam(), loss='mae', metrics=['accuracy'])
        learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7,
                                                                    min_lr=0.0000000003)
        history = model.fit(np.array(x_train), np.array(self.y_train),
                            batch_size=128,
                            epochs=70,
                            validation_data=(x_valid, self.y_valid),
                            callbacks=[learning_rate_reduction])
        # 预测并将结果可视化
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_price = model.predict(x_test)
        model.evaluate(x_test)
        scaler.fit_transform(label.reshape(-1, 1))
        # 反归一化
        closing_price = scaler.inverse_transform(closing_price.reshape(-1, self.pre_stamp)[:, 0].reshape(1, -1))
        y_test = scaler.inverse_transform(np.array(self.y_test).reshape(-1, self.pre_stamp)[:, 0].reshape(1, -1))
        # 计算预测结果
        rms = np.sqrt(np.mean(np.power((y_test[0:1, self.pre_stamp:] - closing_price[0:1, self.pre_stamp:]), 2)))
        print('INFO: 均方根误差 ' + str(rms))
        # loss变化趋势可视化
        plt.figure(figsize=(27, 9))
        plt.subplot(1, 2, 1)
        plt.title('loss_just_stock')
        plt.plot(history.history['loss'], label='training_loss')
        plt.plot(history.history['val_loss'], label='valid_loss')
        plt.legend()
        # 预测效果可视化
        plt.subplot(1, 2, 2)
        plt.title('predict_just_stock')
        plt.plot(closing_price.reshape(1, -1)[0], linewidth=3, alpha=0.8, label='predict_data')
        plt.plot(y_test[0], linewidth=1.2, label='true_data')
        plt.legend()
        plt.show()
        # 保存预测数据
        np.savetxt('./Data/predict_just_stock.csv', closing_price.reshape(1, -1)[0], delimiter=',')

    def tran(self, fileName=[]):
        """训练网络

        Args:
            fileName (list): 读取文件名的列表
        """
        data, label = self.read_data(fileName)
        self.divide_data(data, label)
        self.lstm(label)
        self.lstm_just_stock(label)
