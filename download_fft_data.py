"""
download competition data
"""
import os
from glob import glob
import datetime 
from IPython import embed

datadir = 'data/'
webbase = 'https://oceanofthings.darpa.mil/docs/Sample%20Data/'
# file look like: 'challenge_1-day_sofar_20211109_day8JSON.json'
filebase = 'challenge_1-day_sofar_%s_day%sJSON.json' 
# https://oceanofthings.darpa.mil/docs/Sample%20Data/challenge_1-day_sofar_20211111_day10JSON.json

def download_data():
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    start_day = datetime.date(2021, 11, 2)
    today = datetime.date.today()
    assert start_day < today
    
    for comp_day in range(30):
        data_day = start_day + datetime.timedelta(days=comp_day)
        if data_day <= today:
            fname = filebase%(data_day.strftime('%Y%m%d'), comp_day+1)
            if not os.path.exists(os.path.join(datadir, fname)):
                get_file = webbase + fname
                cmd = 'wget %s -P %s'%(get_file, datadir)
                print(cmd)
                os.system(cmd)
          
if __name__ == '__main__':
    download_data()
