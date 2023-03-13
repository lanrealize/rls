import datetime
import baostock as bs

bs.login()

a = bs.query_history_k_data_plus(code='sh.000001',
                             fields="date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,psTTM,pcfNcfTTM,pbMRQ,isST",
                             start_date='2023-02-01',
                             end_date='2023-03-07',
                                 adjustflag='2')

print()