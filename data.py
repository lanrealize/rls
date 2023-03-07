import baostock as bs
import os
import datetime


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Downloader(object):
    def __init__(self, output_dir='./', date_start='1990-01-01', date_end=None):

        if date_end is None:
            date_end = str(datetime.date.today())

        bs.login()
        self.date_start = date_start
        self.date_end = date_end
        self.output_dir = output_dir
        self.fields = "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date(self.date_end)
        for index, row in stock_df.iterrows():
            print(f'processing {row["code"]} {row["code_name"]}')
            df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end).get_data()
            df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"]}.csv', index=False)


if __name__ == '__main__':
    mkdir('./stockdata/train')
    downloader = Downloader(output_dir='./stockdata/train')
    downloader.run()

    mkdir('./stockdata/test')
    downloader = Downloader(output_dir='./stockdata/test')
    downloader.run()