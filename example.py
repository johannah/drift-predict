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



def load_environment(start_time, download=True, use_gfs=True, use_ncep=False, use_ww3=True, use_pred=True):
    if download:
        download_data()
        download_predictions(DATA_DIR)
    remap_gfs = {'u-component_of_wind_planetary_boundary':'x_wind', 
                 'v-component_of_wind_planetary_boundary':'y_wind'}
    remap_ww3 = { 'Mean_period_of_wind_waves_surface':'sea_surSignificant_height_of_wind_waves_surface', 
                  'Significant_height_of_wind_waves_surface':'sea_surface_wave_significant_height',
                  'u-component_of_wind_surface':'x_wind', 
                  'v-component_of_wind_surface':'y_wind',
                  }
    remap_ncep = {
                  'reftime1':'reftime', 
                  'time1':'time', 
                  'lat':'lat', 
                  'lon':'lon', 
                  'u-component_of_wind_surface':'x_wind', 
                  'v-component_of_wind_surface':'y_wind',
                  }
    #

    readers = []
    # Several models to choose from
    if use_ncep:
        # I think ncep is p5 degree (56km)
        # start: 2021-10-27 06:00:00   end: 2021-11-28 00:00:00   step: 3:00:00
        ncep_wind_data = GenericReader('/Volumes/seahorse/2021-drifters/ncep/Global_Best_v3.nc', standard_name_mapping=remap_ncep)
        readers.append(ncep_wind_data)
    if use_ww3:
        # https://thredds.ucar.edu/thredds/ncss/grib/NCEP/WW3/Global/Best/dataset.html
        ww3_wave_data1 = GenericReader('/Volumes/seahorse/2021-drifters/ww3/Global_Best_1101_1111.nc', standard_name_mapping=remap_ww3)
        # ww3 is only 7 days in advance
        # 2021-11-10 03:00:00   end: 2021-11-28 00:00:00   step: 3:00:00
        ww3_wave_data2 = GenericReader('/Volumes/seahorse/2021-drifters/ww3/Global_Best_1110_1203.nc', standard_name_mapping=remap_ww3)
        readers.append(ww3_wave_data1)
        readers.append(ww3_wave_data2)
    if use_pred:
        weather_files = get_weather(start_time)
        fpattern = 'rtofs_glo_2ds_f'
        last_file = weather_files[-1]
        for nn in weather_files:
            try:
                # readers/basereader/variables.py:                    logger.warning('Assuming time step of 1 hour for ' + self.name)
                print('load', nn)
                basename = os.path.split(nn)[1] 
                # predictions are every hour until 72 hours out
                hours_delta = 1
                if last_file == nn:
                    hours_delta = 10*24 # JUST HOLD THIS LAST FILE FOR A WHILE
                elif basename.startswith(fpattern):
                    fcount = int(basename[len(fpattern):len(fpattern)+3])
                    if fcount >= 72:
                        hours_delta = 3
                r = GenericReader(nn, time_step=datetime.timedelta(hours=hours_delta)) 
                readers.append(r)
            except:
                print('could not load', nn)

    if use_gfs:
        # GFS 0.5 degree (56km)  (higher res than 1 deg)
        # https://thredds.ucar.edu/thredds/gfsp5 # 14 day
        # start: 2021-10-21 00:00:00   end: 2021-12-06 12:00:00   step: 3:00:00
        # gfs should go last in case ww3 is used
 
        gfsp5_wind_data = GenericReader('/Volumes/seahorse/2021-drifters/gfs/Global_0p5deg_Best.nc', standard_name_mapping=remap_gfs)
        readers.append(gfsp5_wind_data)
    return readers

def simulate_spot(spot, start_datetime=None, end_datetime=None, start_at_drifter=False, end_at_drifter=False, plot_plot=False, plot_gif=False):
    spot_df = track_df[track_df['spotterId'] == spot]
    samples = spot_df.index
    ts_col = 'ts_utc'
    timestamps = [x for x in spot_df[ts_col].dt.tz_localize(None)]
    drifter_lons = np.array(spot_df['longitude'])
    drifter_lats = np.array(spot_df['latitude'])
    ot = OceanDrift(loglevel=10)
    [ot.add_reader(r) for r in readers]
    # Prevent mixing elements downwards
    ot.set_config('drift:vertical_mixing', False)
    # TODO fine-tune these
    ot.set_config('drift:horizontal_diffusivity', .01)  # m2/s
    ot.set_config('drift:current_uncertainty', .01)  # m2/s
    ot.set_config('drift:wind_uncertainty', .01)  # m2/s
    start_lon = drifter_lons[0]
    start_lat = drifter_lats[0]
    if start_at_drifter:
        start_datetime = timestamps[0]
    else:
        # no tz in opendrift
        start_datetime = start_datetime.replace(tzinfo=None)
    if end_at_drifter:
        end_datetime = timestamps[-1]
    else:
        end_datetime = end_datetime.replace(tzinfo=None)

    ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                time=start_datetime,
                wind_drift_factor=wind_drift_factor)

    # time step should be in seconds
    ot.run(end_time=end_datetime, time_step_output=datetime.timedelta(hours=1))
    drifter_dict = {'time': timestamps, 'lon': drifter_lons, 'lat': drifter_lats, 
            'label': 'CODE Drifter', 'color': 'b', 'linewidth': 2, 'linestyle':':', 'markersize': 40}
    # Drifter track is shown in red, and simulated trajectories are shown in gray. 
    motion_background = ['x_sea_water_velocity', 'y_sea_water_velocity']
    ot.history.dump(os.path.join(spot_dir, spot))
    if plot_plot:
        try:
            ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), background=motion_background, buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
        except:
            try:
                ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
            except:
                pass
    if plot_gif:
        try:
            ot.animation(filename=os.path.join(spot_dir, '%s.gif'%spot), background=motion_background, buffer=.3, fast=True, drifter=drifter_dict, show_trajectories=True, surface_only=True)
        except:
            pass
if __name__ == '__main__':
    seed = 1110
    test_spots = -1
    future_days = 13
    np.random.seed(seed)
    # ALL TIMES IN UTC
    now = datetime.datetime.now(pytz.UTC)
    now_str = now.strftime("%Y%m%d-%H%M")
    start_time =  datetime.datetime(2021, 11, 17, 17, 0, 0, 0, pytz.UTC)
    end_time =  start_time + datetime.timedelta(days=future_days)
    start_str = start_time.strftime("%Y%m%d-%H%M")
    spot_dir = os.path.join(DATA_DIR, 'results', 'spots_N%s_S%s'%(now_str, start_str))
    if not os.path.exists(spot_dir):
        os.makedirs(spot_dir)
        os.makedirs(os.path.join(spot_dir, 'python'))
        cmd = 'cp *.py %s/' %os.path.join(spot_dir, 'python')
        os.system(cmd)


    # how far do drifters go in 10 days?
    # TODO find wind drift factor
    # TODO measure error
    wind_drift_max = 0.04149
    seed_radius = 100 # in meters
    num_seeds = 100
    wind_drift_factor = np.random.uniform(0, wind_drift_max, num_seeds)

    track_df, wave_df = load_data(search_path='data/challenge*day*JSON.json', start_date=start_time, end_date=end_time)
    spot_names = sorted(track_df['spotterId'].unique())
    # sample a number for testing
    if test_spots > 0:
        spot_names = np.random.choice(spot_names, test_spots)
    print(spot_names)
    readers = load_environment(start_time, download=True, use_gfs=True, use_ncep=False, use_ww3=False, use_pred=True)
    for spot in spot_names:
        #simulate_spot(spot, start_at_drifter=True, end_at_drifter=True, plot_plot=True, plot_gif=False)
        simulate_spot(spot, start_datetime=start_time, end_datetime=end_time, plot_plot=True, plot_gif=False)
        #simulate_spot(spot, start_datetime=start_time, end_at_drifter=True, plot_plot=True, plot_gif=True)


