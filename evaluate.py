import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
from glob import glob
from download_fft_data import download_data
import datetime
import plotly.express as px
from copy import deepcopy
import netCDF4 as nc
import pickle
import time
import pytz
px.set_mapbox_access_token('pk.eyJ1IjoiamgxNzM2IiwiYSI6ImNpaG8wZWNnYjBwcGh0dGx6ZG1mMGl0czAifQ.mhmvIGx34x2fw0s3p9pnaw')

from haversine import haversine, Unit, inverse_haversine, Direction
from IPython import embed
# TRY THIS ONE: https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/
# https://polar.ncep.noaa.gov/waves/viewer.shtml?-multi_1-US_eastcoast-

from opendrift.readers.reader_current_from_drifter import Reader as DrifterReader
from opendrift.readers.reader_current_from_track import Reader as TrackReader
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
from opendrift.readers.reader_netCDF_CF_unstructured import Reader as UnstructuredReader
from opendrift.readers.reader_grib2 import Reader as Grib2Reader
from opendrift.readers.reader_ROMS_native import Reader as ROMSReader
from opendrift.models.openberg import OpenBerg
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories
from utils import DATA_DIR, download_predictions, comp_start_time, comp_end_time, load_environment, make_datetimes_from_args, comp_eval_time
from utils import load_drifter_data
from utils import plot_spot_tracks

# 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'
# //nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.20211120/rtofs
#from mpl_toolkits.basemap import Basemap
#import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
#import netCDF4
# how to load nomads data: 
# https://polar.ncep.noaa.gov/global/examples/usingpython.shtml

def evaluate_spot(spot_df, pred_nc):
    # create random wind drift factors
    samples = spot_df.index
    timestamps = spot_df['ts_utc'] 
    drifter_lons = np.array(spot_df['longitude'])
    drifter_lats = np.array(spot_df['latitude'])
    # Calculate distances (meters) between simulation and synthetic drifter at each time step
    all_pred_times = np.array(pd.to_datetime(pred_nc['time'][:], utc=True, unit='s'))
    pred_times = []
    pred_lons = []
    pred_lats = []
    use_pred = []
    drifter_lons = []
    drifter_lats = []
    drifter_times = []

    eval_time = comp_eval_time
    while eval_time < comp_end_time:
        pred_nearest = np.argmin(abs(all_pred_times-eval_time))
        pred_diff = all_pred_times[pred_nearest] - eval_time


        drift_nearest = spot_df.iloc[spot_df.index.get_loc(eval_time, method='nearest')]
        nearest_ts = drift_nearest['ts_utc']
        drift_diff = nearest_ts - eval_time

        if (abs(pred_diff) < datetime.timedelta(hours=1)) and (abs(drift_diff) < datetime.timedelta(hours=1)):
            drifter_times.append(drift_nearest['ts_utc'])
            drifter_lons.append(drift_nearest['longitude'])
            drifter_lats.append(drift_nearest['latitude'])

            pred_times.append(all_pred_times[pred_nearest])
            pred_lons.append(pred_nc['lon'][:][:,pred_nearest])
            pred_lats.append(pred_nc['lat'][:][:,pred_nearest])

        eval_time = eval_time + datetime.timedelta(days=1) 
    pred_lons = np.array(pred_lons)
    pred_lats = np.array(pred_lats)
    drifter_lons = np.array(drifter_lons)
    drifter_lats = np.array(drifter_lats)
    distances = []
    for s in range(pred_lons.shape[1]):
        distances.append(distance_between_trajectories(pred_lons[:,s], pred_lats[:,s], drifter_lons, drifter_lats))

    return np.array(distances)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir')
    parser.add_argument('--seed', default=1110)
    parser.add_argument('--plot', action='store_true', default=False, help='write plot')
    parser.add_argument('--gif', action='store_true', default=False, help='write gif')
    args = parser.parse_args()
    np.random.seed(args.seed)
    load_dir = args.load_dir
    pred_results = glob(os.path.join(args.load_dir, 'SPOT*.nc'))
    print('found %s predictions'%len(pred_results))
    if len(pred_results):
        load_args = pickle.load(open(os.path.join(args.load_dir, 'args.pkl'), 'rb'))
        start_time, start_str, end_time, end_str = make_datetimes_from_args(load_args)
        #readers = load_environment(start_time, download=False, use_gfs=load_args.use_gfs, use_ncep=load_args.use_ncep, use_ww3=load_args.use_ww3, use_rtofs=load_args.use_rtofs)
        track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json', start_date=start_time)
        spot_distances = {}
        running_sum = []
        for spot_pred_path in pred_results:
            spot = (os.path.split(spot_pred_path)[1]).split('.')[0]
            spot_df = track_df[track_df['spotterId'] == spot]
            spot_nc = nc.Dataset(spot_pred_path)
            distance = evaluate_spot(spot_df, spot_nc)
            min_error = distance.sum(1).min()
            spot_distances[spot] = distance
            print(spot, min_error)
            running_sum.append(min_error)

        print(np.sum(running_sum))
        pickle.dump(spot_distances, open(os.path.join(args.load_dir, 'result_distances.pkl'), 'wb'))


