##
## Created by tioeare on 12/03/22.
##
from getData.getData import Data
from LSTM.lstm import Lstm
from Analysis.analysis import Analysis
from multiprocessing import freeze_support


def main():
    freeze_support()
    stock = Data()
    tran = Lstm()
    ana = Analysis()
    stock.get_data()
    tran.tran(stock.fileName)
    ana.analysis()


if __name__ == '__main__':
    main()
