import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
from glob import glob
from utils import load_data
from utils import plot_spot_tracks
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
from utils import DATA_DIR, get_weather, download_predictions
# 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'
# //nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.20211120/rtofs
#from mpl_toolkits.basemap import Basemap
#import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
#import netCDF4
# how to load nomads data: 
# https://polar.ncep.noaa.gov/global/examples/usingpython.shtml


def wind_drift_spot(spot, ot):
    spot_df = track_df[track_df['spotterId'] == spot]
    samples = spot_df.index
    ts_col = 'ts_utc'
    timestamps = [x for x in spot_df[ts_col].dt.tz_localize(None)]
    drifter_lons = np.array(spot_df['longitude'])
    drifter_lats = np.array(spot_df['latitude'])
    t = ot.get_variables_along_trajectory(variables=['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'],
        lons=drifter_lons, lats=drifter_lats, times=timestamps)
    wind_drift_factor, azimuth = wind_drift_factor_from_trajectory(t)
    print('mean wind drift', spot, wind_drift_factor.mean())
    return wind_drift_factor
    
if __name__ == '__main__':
    from example import load_environment
    seed = 1110
    test_spots = -1
    future_days = 3
    np.random.seed(seed)
    # ALL TIMES IN UTC
    now = datetime.datetime.now(pytz.UTC)
    now_str = now.strftime("%Y%m%d-%H%M")
    start_time =  datetime.datetime(2021, 11, 17, 17, 0, 0, 0, pytz.UTC)
    end_time =  start_time + datetime.timedelta(days=future_days)
    start_str = start_time.strftime("%Y%m%d-%H%M")
    track_df, wave_df = load_data(search_path='data/challenge*day*JSON.json', start_date=start_time, end_date=end_time)
    spot_names = sorted(track_df['spotterId'].unique())
    # sample a number for testing
    if test_spots > 0:
        spot_names = np.random.choice(spot_names, test_spots)
    print(spot_names)
    readers = load_environment(start_time, download=False, use_gfs=True, use_ncep=False, use_ww3=True, use_pred=True)

    ot = OceanDrift(loglevel=1000)
    [ot.add_reader(r) for r in readers]
    # Prevent mixing elements downwards
    ot.set_config('drift:vertical_mixing', False)
    spot_drifts = {}
    np.random.shuffle(spot_names)
    for spot in spot_names[:test_spots]:
        spot_drifts[spot] = wind_drift_spot(spot, ot)
    # found 0.0415
    overall_mean =  np.mean([spot_drifts[spot].mean() for spot in spot_drifts.keys()])
    print('overall mean', overall_mean)
    embed()

