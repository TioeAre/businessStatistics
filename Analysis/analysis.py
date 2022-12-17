##
## Created by tioeare on 12/05/22.
##
import numpy as np


class Analysis(object):
    def __int__(self):
        self.true = []
        self.predict = []
        self.pre_just_stock = []
        print('ok')

    def analysis(self):
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
        print(str(var1) + " " + str(var2))
