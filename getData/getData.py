##
## Created by tioeare on 12/03/22.
##
import json
import copy
from seleniumwire import webdriver
import undetected_chromedriver as uc
import selenium.common.exceptions as exceptions
from bs4 import BeautifulSoup as bs
import time


class Data(object):
    """从雅虎财经网站获取数据

    Attributes:
        params: 五个网站依顺序分别为恒生指数，港元兑人民币，原油价格，黄金价格，美元兑港币
        start_time: 获取数据的起始时间(雅虎财经上港元兑人民币最多只到2003年12月1日)
        end_time: 获取数据的截止时间
        url: 雅虎财经api
    """

    def __init__(self):
        self.params = {
            'stock': '%5EHSI',
            'HKD2CNY': 'HKDCNY=X',
            'CrudeOil': 'CL=F',
            'Gold': 'GC=F',
            'USD2HKD': 'HKD=X'
        }
        self.fileName = ['./Data/stock.json', './Data/Oil.json', './Data/Gold.json', './Data/CNY.json',
                         './Data/USD.json']
        self.start_time = int(time.mktime(time.strptime('2003-12-01 08:00:00', '%Y-%m-%d %H:%M:%S')))
        self.end_time = int(time.mktime(time.strptime('2022-12-03 08:00:00', '%Y-%m-%d %H:%M:%S')))
        self.url = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def save_data(self, url='', fileName='', timestamp=None):
        """保存数据到Data文件夹下

        Args:
            url (str): 五个网站的标识符
            fileName (str): 数据保存位置
            timestamp (list): 恒生指数数据的时间戳列表

        Returns:
            每种数据的json表示
        """
        if timestamp is None:
            timestamp = []
        try:
            chrome_options = webdriver.ChromeOptions()
            driver = uc.Chrome(options=chrome_options)
            driver.get(
                f'{self.url}{url}?formatted=true&crumb=c7O3y0nvz/t&lang=zh-Hant-HK&region=HK&includeAdjustedClose=true&interval=1d&period1={self.start_time}&period2={self.end_time}&events=capitalGain|div|split&useYfid=true&corsDomain=hk.finance.yahoo.com')
            html = driver.page_source
            soup = bs(html, 'lxml')
            cc = soup.select('pre')[0]
            res = json.loads(cc.text)
            data = {}
            # 以下六个分别为时间戳，最低价，最高价，开市价，收市价，经调整收市价
            data['timestamp'] = res['chart']['result'][0]['timestamp']
            if url == self.params['stock']:
                timestamp = copy.deepcopy(data['timestamp'])
            data['low'] = res['chart']['result'][0]['indicators']['quote'][0]['low']
            data['high'] = res['chart']['result'][0]['indicators']['quote'][0]['high']
            data['open'] = res['chart']['result'][0]['indicators']['quote'][0]['open']
            data['close'] = res['chart']['result'][0]['indicators']['quote'][0]['close']
            data['adjclose'] = res['chart']['result'][0]['indicators']['adjclose'][0]['adjclose']
            n = len(data['timestamp'])
            k = len(timestamp)
            i = 0
            while i < k:  # 由于恒生指数的数据和汇率，原油黄金并不是每天都有，并且存在今天这个没有，明天那个没有的情况，以恒生指数为准进行插值
                stock_time = time.strftime("%Y-%m-%d", time.localtime(timestamp[i]))
                this_time = time.strftime("%Y-%m-%d", time.localtime(data['timestamp'][i]))
                data['timestamp'][i] = int(time.mktime(time.strptime(f'{this_time} 08:00:00', '%Y-%m-%d %H:%M:%S')))
                if stock_time == this_time:  # 判断改数据与股票是否有同一天
                    for key in data:  # 判断这一天是否有数据
                        if not data[key][i]:
                            data[key][i] = copy.deepcopy(data[key][i - 1])
                    i += 1
                elif timestamp[i] < data['timestamp'][i]:  # 如果股票有数据而其他没数据则插值
                    n += 1
                    for key in data:
                        data[key].insert(i, copy.deepcopy(data[key][i - 1]))
                    data['timestamp'][i] = copy.deepcopy(timestamp[i])
                else:  # 如果股票没数据而其他有数据则进行删值
                    n -= 1
                    for key in data:
                        del data[key][i]
            if len(data['timestamp']) > k:
                for key in data:
                    del data[key][k]
            with open(fileName, 'w', encoding='utf-8') as f:  # 保存数据为文件
                json.dump(data, f, ensure_ascii=False)
            driver.quit()
            return data
        except IOError:
            print("ERROR: 文件路径出错，请检查或更改")
            raise
        except exceptions.WebDriverException:
            print("ERROR: 浏览器版本错误，请选择108以上的谷歌浏览器. 也可能是网络超时没有挂代理")
            raise

    def get_data(self):
        """从网站获取数据
        """
        timestamp = self.save_data(self.params["stock"], self.fileName[0])
        self.save_data(self.params["CrudeOil"], self.fileName[1], timestamp['timestamp'])
        self.save_data(self.params["Gold"], self.fileName[2], timestamp['timestamp'])
        self.save_data(self.params["HKD2CNY"], self.fileName[3], timestamp['timestamp'])
        self.save_data(self.params["USD2HKD"], self.fileName[4], timestamp['timestamp'])
        print("INFO: Data saved successfully")
