##
## Created by tioeare on 12/05/22.
##
import numpy as np
import json
from scipy import stats

class Analysis(object):
    """
    使用LSTM模型对数据进行训练

    Attributes:
        true: 真实值列表
        predict: 预测值列表
        pre_just_stock: 无影响因素的预测值列表
        result: 分析结果字典
    """
    def __int__(self):
        self.true = []
        self.predict = []
        self.pre_just_stock = []

    def analysis(self):
        """分析数据
        """
        with open("./Data/true.csv", encoding='utf-8') as f:
            self.true = np.loadtxt("./Data/true.csv")
        with open("./Data/predict.csv", encoding='utf-8') as f:
            self.predict = np.loadtxt("./Data/predict.csv")
        with open("./Data/predict_just_stock.csv", encoding='utf-8') as f:
            self.pre_just_stock = np.loadtxt("./Data/predict_just_stock.csv")
        tss1 = tss2 = 0
        for i in range(self.true.shape[0]):
            tss1 += pow((self.true[i] - self.predict[i]), 2)
            tss2 += pow((self.true[i] - self.pre_just_stock[i]), 2)
        var1 = tss1 / self.true.shape[0]
        var2 = tss2 / self.true.shape[0]
        t_1, p_1 = stats.ttest_ind(self.true, self.predict)
        t_2, p_2 = stats.ttest_ind(self.true, self.pre_just_stock)
        self.result = {}
        self.result['mean'] = {'true': round(float(np.mean(self.true)), 4), 'predict': round(float(np.mean(self.predict)), 4),
                               'pre_just_stock': round(float(np.mean(self.pre_just_stock)), 4)}
        self.result['var'] = {'true': round(float(np.var(self.true)), 4), 'predict': round(float(np.var(self.predict)), 4),
                               'pre_just_stock': round(float(np.var(self.pre_just_stock)), 4)}
        self.result['mean square error'] = {'predict': round(var1, 4), 'pre_just_stock': round(var1, 4)}
        self.result['t'] = {'predict': round(t_1, 4), 'pre_just_stock': round(t_2, 4)}
        self.result['p'] = {'predict': round(p_1, 9), 'pre_just_stock': round(p_2, 4)}
        with open('./Data/analysis.json', 'w', encoding='utf-8') as f:
            json.dump(self.result, f, ensure_ascii=False)
