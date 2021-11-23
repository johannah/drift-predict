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
from opendrift.models.leeway import Leeway
from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories
from utils import DATA_DIR, download_predictions, load_environment, make_datetimes_from_args
from utils import load_drifter_data, plot_spot_tracks

def simulate_spot(spot, start_datetime=None, end_datetime=None, start_at_drifter=False, end_at_drifter=False, plot_plot=False, plot_gif=False, num_seeds=100, seed_radius=10, wind_drift_factor_max=.02, model_type='OceanDrift', object_type=26):
    # create random wind drift factors
    # mean wind drift factor is found to be 0.041
    # min wind drift factor is found to be 0.014
    # max wind drift factor is found to be 0.16
    spot_df = track_df[track_df['spotterId'] == spot]
    samples = spot_df.index
    ts_col = 'ts_utc'
    timestamps = [x for x in spot_df[ts_col].dt.tz_localize(None)]
    drifter_lons = np.array(spot_df['longitude'])
    drifter_lats = np.array(spot_df['latitude'])
    if model_type == 'OceanDrift':
        ot = OceanDrift(loglevel=80) # lower log is more verbose
        # Prevent mixing elements downwards
        ot.set_config('drift:vertical_mixing', False)
    if model_type == 'Leeway':
        ot = Leeway(loglevel=80)
    [ot.add_reader(r) for r in readers]
    # TODO fine-tune these. 0.01 seemed too small
    ot.set_config('drift:horizontal_diffusivity', .1)  # m2/s
    ot.set_config('drift:current_uncertainty', .1)  # m2/s
    ot.set_config('drift:wind_uncertainty', .1)  # m2/s
    # find nearest timestep to start
    if start_at_drifter:
        start_datetime = timestamps[0]
    else:
        # no tz in opendrift
        start_datetime = start_datetime.replace(tzinfo=None)
    if end_at_drifter:
        end_datetime = timestamps[-1]
    else:
        end_datetime = end_datetime.replace(tzinfo=None)

    # seed from time nearest to start time's location of drifters
    diff_time = abs(start_time-spot_df.index)
    drift_ts_index = np.argmin(diff_time)
    drift_ts = spot_df.index[drift_ts_index]
    if np.abs(start_time-drift_ts) > datetime.timedelta(hours=4):
        print("NO NEAR TIME DRIFTER", drift_ts, spot_df.loc[drift_ts]['spotterId'])
        return
    if end_datetime < start_datetime: 
        print('ending before starting')
        return
    try:
        start_lon = spot_df.loc[drift_ts]['longitude']
        start_lat = spot_df.loc[drift_ts]['latitude']
        if model_type == 'OceanDrift':
            wind_drift_factor = np.linspace(0.001, wind_drift_factor_max, num_seeds)
            ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                                   time=start_time.replace(tzinfo=None),
                                   wind_drift_factor=wind_drift_factor)
            variables = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind', 'sea_surface_wave_significant_height', 'sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment']
        # time step should be in seconds
        elif model_type == 'Leeway':
            ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                                   time=start_time.replace(tzinfo=None),
                                   object_type=object_type)
            variables = ['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind']
        ot.run(end_time=end_datetime.replace(tzinfo=None), time_step=datetime.timedelta(hours=1), 
               time_step_output=datetime.timedelta(hours=1), outfile=os.path.join(spot_dir, spot + '.nc'))

        drifter_dict = {'time': timestamps, 'lon': drifter_lons, 'lat': drifter_lats, 
                'label': 'CODE Drifter', 'color': 'b', 'linewidth': 2, 'linestyle':':', 'markersize': 40}
        # Drifter track is shown in red, and simulated trajectories are shown in gray. 
        motion_background = ['x_sea_water_velocity', 'y_sea_water_velocity']
        ot.history.dump(os.path.join(spot_dir, spot+'.npy'))
        # save drift track
        #along_track = ot.get_variables_along_trajectory(variables=variables, lons=drifter_lons, lats=drifter_lats, times=timestamps)
        #pickle.dump(along_track, open(os.path.join(spot_dir, spot+'_drifter_thru_model.pkl'), 'wb'))
    except Exception as e:
        print(e)
    if plot_plot:
        #try:
        #    ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), background=motion_background, buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
        #except:
        try:
            ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
        except:
            try:
                ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), buffer=.01, fast=True, cmap='viridis')
            except:
                pass
    if plot_gif:
        try:
            ot.animation(filename=os.path.join(spot_dir, '%s.gif'%spot), background=motion_background, buffer=.3, fast=True, drifter=drifter_dict, show_trajectories=True, surface_only=True)
        except:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1110)
    parser.add_argument('--load-dir', default='', help='load partially-complete experiment from this dir')
    parser.add_argument('--model-type', default='OceanDrift', help='type of model', choices=['OceanDrift', 'Leeway'])
    parser.add_argument('--object-type', default=70, type=int, help='type of model 72 is oil drum, best result', choices=[69, 70, 71, 72]) # bait boxes

    parser.add_argument('--num-seeds', default=100, type=int, help='num particles to simulate')
    parser.add_argument('--seed-radius', default=500, type=int, help='meters squared region to seed particles in simulation')
    parser.add_argument('--wind-drift-factor-max', '-wdm', default=0.06, type=float, help='max wind drift factor to use when seeding particles. default was found experimentally with get_wind_drift_factor.py')
    parser.add_argument('--start-year', default=2021, type=int)
    parser.add_argument('--start-month', default=11, type=int)
    parser.add_argument('--start-day', default=17, type=int)
    parser.add_argument('--start-hour', default=17, type=int)
    parser.add_argument('--future-days', '-fd', default=16, type=int)
    parser.add_argument('--test-spots', default=-1, help='number of random spots to run. if negative, all spots will be evaluated')
    parser.add_argument('--download', action='store_true', default=False, help='download new data')
    parser.add_argument('--start-at-drifter', '-sd', action='store_true', default=False, help='start simulation at drifter start')
    parser.add_argument('--end-at-drifter', '-ed', action='store_true', default=False, help='end simulation at drifter start')
    parser.add_argument('--plot', action='store_true', default=False, help='write plot')
    parser.add_argument('--gif', action='store_true', default=False, help='write gif')
    parser.add_argument('--use-ncss', '-n', action='store_true', default=False, help='include ncss data - current data. download manually to DATA_DIR/ncss/')
    parser.add_argument('--use-ww3', '-w', action='store_true', default=False, help='include ww3 - 8 day wave forecast. download manually to DATA_DIR/ww3/')
    parser.add_argument('--use-gfs', '-g', action='store_true', default=False, help='include gfs - 14 day wind forecast. download manually to DATA_DIR/gfs/')
    parser.add_argument('--use-rtofs', '-r', action='store_true', default=False, help='include rtofs currents (auto download 7 day forecasts to download manually to DATA_DIR/pred/)')
    args = parser.parse_args()
    now = datetime.datetime.now(pytz.UTC)
    # ALL TIMES IN UTC

    load_from_dir = ''
    if args.load_dir != '':
        load_from_dir = args.load_dir
        spot_dir = args.load_dir
        # reload w same args
        args = pickle.load( open(os.path.join(spot_dir, 'args.pkl'), 'rb'))
    np.random.seed(args.seed)
    now_str = now.strftime("%Y%m%d-%H%M")
    start_time, start_str, end_time, end_str = make_datetimes_from_args(args)
    if load_from_dir ==  '':
        if args.model_type == 'Leeway': 
            model_name = args.model_type + str(args.object_type)
        else:
            model_name = 'WD%.02f_'%(args.wind_drift_factor_max) + args.model_type 
        spot_dir = os.path.join(DATA_DIR, 'results', 'spots_N%s_S%s_E%s_DS%s_DE%s_R%sG%sW%sN%s_%s'%(now_str, 
                                     start_str, end_str, int(args.start_at_drifter), int(args.end_at_drifter), 
                                     int(args.use_rtofs), int(args.use_gfs), int(args.use_ww3), int(args.use_ncss), 
                                     model_name))
        if not os.path.exists(spot_dir):
            os.makedirs(spot_dir)
            os.makedirs(os.path.join(spot_dir, 'python'))
            cmd = 'cp *.py %s/' %os.path.join(spot_dir, 'python')
            os.system(cmd)
            pickle.dump(args, open(os.path.join(spot_dir, 'args.pkl'), 'wb'))


    # how far do drifters go in 10 days?
    # TODO find wind drift factor
    # TODO measure error

    #track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json', start_date=start_time, end_date=end_time)
    track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json')
    spot_names = sorted(track_df['spotterId'].unique())
    # sample a number for testing
    if args.test_spots > 0:
        spot_names = np.random.choice(spot_names, args.test_spots)
    print(spot_names)
    readers = load_environment(start_time, download=args.download, use_gfs=args.use_gfs, use_ncss=args.use_ncss, use_ww3=args.use_ww3, use_rtofs=args.use_rtofs)
    for spot in spot_names:
        if not os.path.exists(os.path.join(spot_dir, spot + '.nc')):
            simulate_spot(spot, start_datetime=start_time,  end_datetime=end_time,\
                          start_at_drifter=args.start_at_drifter,  end_at_drifter=args.end_at_drifter, \
                          plot_plot=args.plot, plot_gif=args.gif, num_seeds=args.num_seeds, 
                          seed_radius=args.seed_radius, wind_drift_factor_max=args.wind_drift_factor_max, 
                          model_type=args.model_type, object_type=args.object_type)


