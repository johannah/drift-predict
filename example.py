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
px.set_mapbox_access_token('pk.eyJ1IjoiamgxNzM2IiwiYSI6ImNpaG8wZWNnYjBwcGh0dGx6ZG1mMGl0czAifQ.mhmvIGx34x2fw0s3p9pnaw')

from haversine import haversine, Unit, inverse_haversine, Direction
from IPython import embed
# TRY THIS ONE: https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/
# https://polar.ncep.noaa.gov/waves/viewer.shtml?-multi_1-US_eastcoast-
# download_data()

from opendrift.readers.reader_current_from_drifter import Reader as DrifterReader
from opendrift.readers.reader_current_from_track import Reader as TrackReader
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
from opendrift.readers.reader_netCDF_CF_unstructured import Reader as UnstructuredReader
from opendrift.readers.reader_grib2 import Reader as Grib2Reader
from opendrift.readers.reader_ROMS_native import Reader as ROMSReader
from opendrift.models.openberg import OpenBerg
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories
from utils import DATA_DIR, get_weather
# 'https://tds.hycom.org/thredds/dodsC/GLBy0.08/latest'
# //nomads.ncep.noaa.gov/pub/data/nccf/com/rtofs/prod/rtofs.20211120/rtofs
#from mpl_toolkits.basemap import Basemap
#import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
#import netCDF4
# how to load nomads data: 
# https://polar.ncep.noaa.gov/global/examples/usingpython.shtml


# how far do drifters go in 10 days?
# TODO find wind drift factor
# TODO measure error
# TODO wtf is up with the longitude being so weird - 
wind_drift_max = 0.033
seed_radius = 10 # in meters
num_seeds = 10
wind_drift_factor = np.random.uniform(0, wind_drift_max, num_seeds)
#track_df, wave_df = load_data(search_path='data/challenge_*day18JSON.json')

remap_ncss = {
              'reftime1':'reftime', 
              'time1':'time', 
              'lat':'lat', 
              'lon':'lon', 
              'Wind_direction_from_which_blowing_surface':'wind_from_direction',
              'Wind_speed_surface':'wind_speed', 
              }

               #'Direction_of_wind_waves_surface':'', 
              #'LatLon_Projection':'LatLon_Projection', 
              #'Mean_period_of_wind_waves_surface':'', 
              #'Primary_wave_direction_surface':'', 
              #'Primary_wave_mean_period_surface':'', 
              #'Significant_height_of_combined_wind_waves_and_swell_surface':'', 
              #'Significant_height_of_wind_waves_surface':'',
              #'u-component_of_wind_surface':'' ,
              #'v-component_of_wind_surface':'', 
              #'Direction_of_swell_waves_ordered_sequence_of_data':'', 
              #'ordered_sequence_of_data':'', 
              #'Mean_period_of_swell_waves_ordered_sequence_of_data':''}


#d = GenericReader('/Volumes/seahorse/2021-drifters/ncep/Global_Best.nc', standard_name_mapping=remap_ncss)

track_df, wave_df = load_data(search_path='data/challenge*202111*1*day*JSON.json')
start_time =  datetime.datetime(2021, 11, 18, 0, 0, 0)
weather_files = get_weather(start_time)
#track_df, wave_df = load_data(search_path='data/challenge_2021*day*JSON.json')
spot_names = sorted(track_df['spotterId'].unique())
print(spot_names)
#spot_extent_dict, extent_dict = make_weather_clips(spot_names, track_df)
#for spot, extent in spot_extent_dict.items():
readers = [GenericReader(nn) for nn  in weather_files]
for spot in spot_names:
    spot_df = track_df[track_df['spotterId'] == spot]
    samples = spot_df.index
    ts_col = 'ts_utc'
    timestamps = [x for x in spot_df[ts_col].dt.tz_localize(None)]
    drifter_lons = np.array(spot_df['longitude'])
    drifter_lats = np.array(spot_df['latitude'])
    ot = OceanDrift(loglevel=10)
    #r = DrifterReader(
    #    lons=drifter_lons, lats=drifter_lats, times=timestamps)
    #ot.add_reader(r)
    # DEBUG 
    [ot.add_reader(r) for r in readers]
    #[ot.add_reader(r) for r in readers]
    #ot.add_readers_from_list(load_nc, lazy=False)
    # Prevent mixing elements downwards
    ot.set_config('drift:vertical_mixing', False)

    # TODO UTC or East?
    start_sample = samples[0]
    end_sample = samples[-1]
    start_time = timestamps[0]
    end_time =  timestamps[-1]
    # TODO find start 
    start_lon = drifter_lons[0]
    start_lat = drifter_lats[0]
    # TODO how to do start time
    #time=reader_current.start_time,
    ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                time=readers[0].start_time,
                wind_drift_factor=wind_drift_factor)

    # time step should be in seconds
    #ot.run(end_time=r.end_time)
    ot.run(end_time=readers[-1].end_time)
    drifter_dict = {'time': timestamps, 'lon': drifter_lons, 'lat': drifter_lats, 
            'label': 'CODE Drifter', 'color': 'b', 'linewidth': 2, 'linestyle':':', 'markersize': 40}
    # Drifter track is shown in red, and simulated trajectories are shown in gray. 
    motion_background = ['x_sea_water_velocity', 'y_sea_water_velocity']
    #ot.plot(filename='example.png', buffer=.01, fast=True, cmap='viridis',
    #        trajectory_dict={
    #        'lon':drifter_lons, 'lat':drifter_lats,
    #        'time':timestamps, 'linestyle': 'r-'})
    np.save(spot, ot.history)
    try:
        ot.plot(filename='%s_example.png'%spot, buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
        ot.animation(filename='%s_example.gif'%spot,  buffer=.01, fast=True, drifter=drifter_dict, show_trajectories=True, surface_only=True)
    except:
        pass
    # ot.history['lat']
    # ot.history['sea_water_velocity_u']

embed()

