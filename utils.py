import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

import datetime
import numpy as np
import pandas as pd
from glob import glob
import os
import sys
import pytz
from IPython import embed; 

DATA_DIR = '/Volumes/seahorse/2021-drifters/'
comp_start_time = datetime.datetime(2021, 11, 1, 17, 0, 0, 0, pytz.UTC)
comp_end_time = comp_start_time + datetime.timedelta(days=10)

def get_weather(start_time=comp_start_time):
    pred_dir = os.path.join(DATA_DIR, 'pred')
    avail_dates = glob(os.path.join(pred_dir, '2021*'))
    # for days older than today, use nowcast date 
    today = pytz.utc.localize(datetime.datetime.utcnow())
    assert start_time <= today
    date = start_time
    weather_files = []
    today_str = "%s%s%s"%(today.year, today.month, today.day)
    while date < today:
        date_str = "%s%s%s"%(date.year, date.month, date.day)
        # get hindcast data
        if date <= today:
            print('getting hindcast data for %s'%date)
            search_path = os.path.join(pred_dir, date_str, 'rtofs_glo_2ds_n0*progremap.nc')
            files = glob(search_path)
            print('found %s files' %len(files))
            if not len(files):
                print("WARNING no files found")
                print(search_path)
            weather_files.extend(sorted(files))
        date = date+ datetime.timedelta(days=1)

    # try looking for today data 
    search_path = os.path.join(pred_dir, today_str, 'rtofs_glo_2ds_f*progremap.nc')
    files = glob(search_path)
    print('found %s today files' %len(files))
    if not len(files):
        print('using yesterdays forecast data')
        yes = today + datetime.timedelta(days=-1)
        yes_str = "%s%s%s"%(yes.year, yes.month, yes.day)
        search_path = os.path.join(pred_dir, yes_str, 'rtofs_glo_2ds_*progremap.nc')
        files = glob(search_path)
        print('found %s yesteday files' %len(files))
    weather_files.extend(sorted(files))
    return weather_files
 
    
def make_weather_clips(spot_names, df):
    # NOT NEEDED
    extents = []
    spot_dict = {}
    data = {}
    for sc, spot in enumerate(spot_names):
        data[spot] = df[df['spotterId'] == spot]
        if not data[spot].shape[0]:
            print('spot  not available', spot)
        else:
            # original .nc files are too large to be loaded, make a small one for this spot
            # To select the region with the longitudes from 120E to 90W and latitudes from 20N to 20S from all input fields in ifile and write the result to ofile 
            # cdo sellonlatbox,120,-90,20,-20 ifile ofile
            # cdo sellonlatbox,4.5,5.4,51.5,52.45 model_fc.nc model_fc_slice.nc
            # TODO this does not handle edge cases
            min_lat = np.round(data[spot]['latitude'].min() - 1)
            max_lat = np.round(data[spot]['latitude'].max() + 1)
            min_lon = np.round(data[spot]['longitude'].min() - 1)
            max_lon = np.round(data[spot]['longitude'].max() + 1)
            extents.append((min_lat, max_lat, min_lon, max_lon))
            spot_dict[spot] = extents[-1]
    unique_extents = set(extents)
    print('have %s unique extents' %len(unique_extents))
    download_predictions(DATA_DIR)
    pred_dir = os.path.join(DATA_DIR, 'pred')
    dates = glob(os.path.join(pred_dir, '2021*'))
    extent_dict = {}
    for e in unique_extents:
        extent_dict[e] = []
    print('found dates', dates)
    # make prediction files
    for date in dates:  
        clip_dir = os.path.join(date, 'clips')
        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        preds = glob(os.path.join(date, '*.nc'))
        for pred in preds: 
            pred_name = os.path.split(pred)[1]
            for extent in unique_extents:
                min_lat, max_lat, min_lon, max_lon = extent
                ll = "_clip_{0}_{1}_{2}_{3}.nc".format(min_lat, max_lat, min_lon, max_lon)
                clip_fpath = os.path.join(clip_dir, pred_name.replace('.nc', ll))
                extent_dict[extent].append(clip_fpath)
                if not os.path.exists(clip_fpath):
                    print('missing', clip_fpath)
                    cdo_cmd = "cdo sellonlatbox,{0},{1},{2},{3} {4} {5}".format(min_lat, max_lat, min_lon, max_lon, pred, clip_fpath)
                    os.system(cdo_cmd)
    return spot_dict, extent_dict

def download_predictions(data_dir):
    """
    Download the RTOFS forecast
    
    Home > RTOFS model forecast documentation
    RTOFS (Atlantic) is a basin-scale ocean forecast system based on the HYbrid Coordinate Ocean Model (HYCOM).
    
    The model is run once a day, completing at about 1400Z. Each run starts with a 24 hour assimiliation hindcast and produces ocean surface forecasts every hour and full volume forecasts every 24 hours from the 0000Z nowcast out to 120 hours.
    
    For example for the 20211118 model data there are 138 files:
    ** future **
    20211118/rtofs_glo_2ds_f003_prog.nc
    - 2021-11-18 03:00:00
    20211118/rtofs_glo_2ds_f023_prog.nc
    -  2021-11-18 23:00:00
    20211118/rtofs_glo_2ds_f123_prog.nc
    - 2021-11-23 03:00:00
    ** nowcast (hindcast) **
    20211118/rtofs_glo_2ds_n014_prog.nc
    - 2021-11-17 14:00:00

    Latitude-longitude points on the native grid for the Atlantic RTOFS model are on a curvilinear orthogonal grid. The latitude-longitude point files for correct interpretation of the data files as well as other data files and software for processing Atlantic RTOFS data is available here. See the README and the Readme.RTOFS files for more information on the files and software. Vertical coordinates are interpolated to 40 fixed-depth positions, as specified here.

# prog files look like:
root group (NETCDF4 data model, file format HDF5):
    Conventions: CF-1.0
    title: HYCOM ATLb2.00
    institution: National Centers for Environmental Prediction
    source: HYCOM archive file
    experiment: 92.8
    history: archv2ncdf2d
    dimensions(sizes): MT(1), Y(3298), X(4500), Layer(1)
    variables(dimensions): float64 MT(MT), float64 Date(MT), int32 Layer(Layer), int32 Y(Y), int32 X(X), float32 Latitude(Y, X), float32 Longitude(Y, X), float32 u_velocity(MT, Layer, Y, X), float32 v_velocity(MT, Layer, Y, X), float32 sst(MT, Y, X), float32 sss(MT, Y, X), float32 layer_density(MT, Layer, Y, X)
    groups:

    """
    today = datetime.datetime.now()
    yes = today + datetime.timedelta(days=-1)
    tom = today + datetime.timedelta(days=1)
    for date in [yes, today, tom]:
        date_str = "%s%s%s"%(date.year, date.month, date.day)
        print('starting date', date)
        pred_dir = os.path.join(data_dir, 'pred', date_str)
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        # search for old ls files
        old_files = glob(os.path.join(pred_dir, 'ls-l*'))
        for old in old_files:
            os.remove(old)

        wget_ls = 'wget https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.%s/ls-l -P %s'%(date_str, pred_dir)
        os.system(wget_ls)
        ls_fpath = os.path.join(pred_dir, 'ls-l')
        # if day is ready, read it
        if os.path.exists(ls_fpath):
            ls = open(ls_fpath, 'r')
            get_files = []
            for ff in ls:
                if 'prog.nc' in ff:
                    # TODO prefer nowcast
                    # rtofs_glo_2ds_f000_prog.nc # future
                    # rtofs_glo_2ds_n000_prog.nc # nowcast
                    get_files.append(ff.split(' ')[-1].strip())
                    target_path = os.path.join(pred_dir, get_files[-1]) 
                    print(target_path)
                    if not os.path.exists(target_path):
                        wget_ls = 'wget https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.%s/%s -P %s'%(date_str, get_files[-1], pred_dir)
                        print('downloading', get_files[-1])
                        os.system(wget_ls)
                    nn_path = target_path.replace('.nc', 'remap.nc')
                    if not os.path.exists(nn_path):
                        # remap colinear - should check this!
                        cmd = 'cdo remapnn,global_.08 %s %s'%(target_path, nn_path)
                        os.system(cmd)

def load_data(search_path='data/challenge_*day*.json', start_date=comp_start_time, end_date=comp_end_time):
    # load sorted dates
    dates = sorted(glob(search_path))
    bad_spots = []
    track_columns = ['latitude', 'longitude', 'timestamp', 'spotterId', 'day', 'date']
    wave_columns = ['significantWaveHeight', 'peakPeriod', 'meanPeriod', 'peakDirection',
                                                  'peakDirectionalSpread', 'meanDirection', 'meanDirectionalSpread',
                                                  'timestamp', 'latitude', 'longitude', 'spotterId', 'day', 'date']
    wave_df = pd.DataFrame(columns=wave_columns)
    track_df = pd.DataFrame(columns=track_columns)
    start_day = start_date+datetime.timedelta(days=-2)
    for cnt, date in enumerate(dates):  
        st = date.index('sofar')+len('sofar_')
        en = st + 8
        date_str = date[st:en]
        date_ts = datetime.datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]), 0, 0, 0, 0, pytz.UTC)
        if date_ts >= start_day:
            print(date_ts, start_day)
            print('loading date: %s'% date)
            day_data = json.load(open(date))['all_data']
            for spot in range(len(day_data)):
                spot_day_data = day_data[spot]['data']
                this_track_df = pd.DataFrame(spot_day_data['track'])
                this_wave_df = pd.DataFrame(spot_day_data['waves'])
                this_track_df['spotterId'] = spot_day_data['spotterId']
                this_wave_df['spotterId'] = spot_day_data['spotterId']
                this_track_df['day'] = cnt
                this_wave_df['day'] = cnt
                # get date from filename
                this_track_df['date'] = date_str 
                this_wave_df['date'] = date_str
                if len(this_track_df.columns) != len(track_columns):
                    # some spots have no data
                    bad_spots.append((date, spot, spot_day_data))
                else:
                    track_df = track_df.append(this_track_df)
 
                if len(this_wave_df.columns) != len(wave_columns):
                    # some spots have no data
                    bad_spots.append((date, spot, spot_day_data))
                else:
                    wave_df = wave_df.append(this_wave_df)
    track_df = track_df.drop_duplicates()
    track_df['real_sample'] =  1 
    track_df['sample_num'] = 1
    track_df.index = np.arange(track_df.shape[0])
    for spot in track_df['spotterId'].unique():
        spot_indexes = track_df[track_df['spotterId'] == spot].index
        track_df.loc[spot_indexes, 'sample_num']  = np.arange(len(spot_indexes), dtype=np.int64)
    track_df['scaled_sample_num'] = track_df['sample_num'] / track_df['sample_num'].max() 
    track_df['ts'] = track_df['timestamp']
    track_df['ts_utc'] = pd.to_datetime(track_df['ts'])
    track_df.index = track_df['ts_utc']
    #track_df['ts_east'] = track_df['ts_utc']
    #track_df = track_df.tz_convert("US/Eastern")

    # sometimes there are multiple entries for same ts
    wave_df = wave_df.drop_duplicates()
    wave_df.index = np.arange(wave_df.shape[0])
    wave_df['real_sample'] =  1 
    wave_df['ts'] = wave_df['timestamp']
    wave_df['ts_utc'] = pd.to_datetime(wave_df['ts'])
    wave_df.index = wave_df['ts_utc']
    #wave_df['ts_east'] = wave_df['ts_utc']
    #wave_df = wave_df.tz_convert("US/Eastern")
    track_df = track_df[track_df['ts_utc'] > start_date]
    wave_df = wave_df[wave_df['ts_utc'] > start_date]

    track_df = track_df[track_df['ts_utc'] < end_date]
    wave_df = wave_df[wave_df['ts_utc'] < end_date]
    return track_df, wave_df

def plot_spot_tracks(track_df, savedir='spot_plots'):
    for spot in track_df['spotterId'].unique():
        spot_track_df = track_df[track_df['spotterId'] == spot]
        plt.figure()
        sns.scatterplot(x="longitude", y="latitude",
                        hue="spotterId", size='date',
                        data=spot_track_df)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, '%s.png'%spot))
        plt.close()

if __name__ == '__main__':
    #load_data(search_path='data/challenge_*day*.json')
    download_predictions(data_dir=DATA_DIR)

