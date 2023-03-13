import baostock as bs
import os
import datetime
from tqdm import tqdm


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
        self.fields = "date,code,open,high,low,close,volume,amount,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST"

    def get_codes_by_date(self):
        one_day = datetime.timedelta(days=1)
        today = datetime.date.today()
        yesterday = today - one_day
        stock_rs = bs.query_all_stock(yesterday)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        stock_df = self.get_codes_by_date()
        for index, row in tqdm(stock_df.iterrows(), total=stock_df.shape[0]):
            if 'sh' in row["code"]:
                if os.path.exists(f'{self.output_dir}/{row["code"]}.{row["code_name"]}.csv'.replace('*', '')):
                    print(f'{row["code"]} {row["code_name"]} exist')
                    continue
                print(f'processing {row["code"]} {row["code_name"]}')
                df_code = bs.query_history_k_data_plus(row["code"], self.fields,
                                                       start_date=self.date_start,
                                                       end_date=self.date_end, adjustflag='2').get_data()
                df_code.to_csv(f'{self.output_dir}/{row["code"]}.{row["code_name"]}.csv'.replace('*', ''), index=False)


if __name__ == '__main__':
    mkdir('D:/stockdata/train')
    downloader = Downloader(output_dir='D:/stockdata/train', date_start='1990-01-01', date_end='2023-01-31')
    downloader.run()

    mkdir('D:/stockdata/test')
    downloader = Downloader(output_dir='D:/stockdata/test', date_start='2023-02-01', date_end='2023-03-07')
    downloader.run()


mkdir('D:/stockdata/train')
downloader = Downloader(output_dir='D:/stockdata/train', date_start='1990-01-01', date_end='2023-01-31')
downloader.run()

mkdir('D:/stockdata/test')
downloader = Downloader(output_dir='D:/stockdata/test', date_start='2023-02-01', date_end='2023-03-07')
downloader.run()
