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

from haversine import haversine, Unit, inverse_haversine, Direction
from IPython import embed

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
from utils import load_environment_data, make_datetimes_from_args
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
        ot = Leeway(loglevel=20)
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
    if np.abs(start_time-drift_ts) > datetime.timedelta(hours=1):
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
        # time step should be in seconds
        if model_type == 'Leeway':
            ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                                   time=start_time.replace(tzinfo=None),
                                   object_type=object_type)
        ot.run(end_time=end_datetime.replace(tzinfo=None), time_step=datetime.timedelta(hours=1),
               time_step_output=datetime.timedelta(hours=1), outfile=os.path.join(spot_dir, spot + '.nc'))
        drifter_dict = {'time': timestamps, 'lon': drifter_lons, 'lat': drifter_lats,
                'label': 'CODE Drifter', 'color': 'b', 'linewidth': 2, 'linestyle':':', 'markersize': 40}
        # Drifter track is shown in red, and simulated trajectories are shown in gray.
        motion_background = ['x_sea_water_velocity', 'y_sea_water_velocity']
        ot.history.dump(os.path.join(spot_dir, spot+'.npy'))
    except Exception as e:
        print(e)
        embed()
    if plot_plot:
        #try:
        #    ot.plot(filename=os.path.join(spot_dir, '%s.png'%spot), background=motion_background, buffer=.01, fast=True, cmap='viridis',  trajectory_dict=drifter_dict)
        #except:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1110)
    parser.add_argument('--load-dir', default='', help='load partially-complete experiment from this dir')
    parser.add_argument('--data-dir', default='data', help='load environmental and drifter data from this dir')
    parser.add_argument('--save-dir', default='results', help='save results in this dir')
    parser.add_argument('--model-type', default='Leeway', help='type of model', choices=['OceanDrift', 'Leeway'])
    parser.add_argument('--object-type', default=70, help='type of model', choices=[69, 70, 71, 72]) # bait boxes
    parser.add_argument('--num-seeds', default=100, type=int, help='num particles to simulate')
    parser.add_argument('--seed-radius', default=500, type=int, help='meters squared region to seed particles in simulation')
    parser.add_argument('--wind-drift-factor-max', '-wdm', default=0.06, type=float, help='max wind drift factor to use when seeding particles. default was found experimentally with get_wind_drift_factor.py')
    parser.add_argument('--start-year', default=2021, type=int)
    parser.add_argument('--start-month', default=11, type=int)
    parser.add_argument('--start-day', default=22, type=int)
    parser.add_argument('--start-hour', default=17, type=int)
    parser.add_argument('--future-days', '-fd', default=6, type=int)
    parser.add_argument('--test-spots', default=-1, help='number of random spots to run. if negative, all spots will be evaluated')
    parser.add_argument('--start-at-drifter', '-sd', action='store_true', default=False, help='start simulation at drifter start')
    parser.add_argument('--end-at-drifter', '-ed', action='store_true', default=False, help='end simulation at drifter start')
    parser.add_argument('--plot', action='store_true', default=False, help='write plot')
    parser.add_argument('--gif', action='store_true', default=False, help='write gif')
    parser.add_argument('--use-ncep', '-n', action='store_true', default=False, help='include ncep data - wind data')
    parser.add_argument('--use-ww3', '-w', action='store_true', default=False, help='include ww3 - 8 day wave forecast')
    parser.add_argument('--use-gfs', '-g', action='store_true', default=False, help='include gfs - 14 day wind forecast')
    parser.add_argument('--use-rtofs', '-r', action='store_true', default=False, help='include rtofs current forecasts')
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
        spot_dir = os.path.join(args.save_dir, 'spots_N%s_S%s_E%s_DS%s_DE%s_R%sG%sW%sN%s_%s'%(now_str,
                                     start_str, end_str, int(args.start_at_drifter), int(args.end_at_drifter),
                                     int(args.use_rtofs), int(args.use_gfs), int(args.use_ww3), int(args.use_ncep),
                                     model_name))
        if not os.path.exists(spot_dir):
            os.makedirs(spot_dir)
            os.makedirs(os.path.join(spot_dir, 'python'))
            cmd = 'cp *.py %s/' %os.path.join(spot_dir, 'python')
            os.system(cmd)
            pickle.dump(args, open(os.path.join(spot_dir, 'args.pkl'), 'wb'))


    #track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json', start_date=start_time, end_date=end_time)
    #track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json')
    track_df = pd.read_csv(os.path.join(args.data_dir, 'challenge_30-day_sofar_20211102_csv.csv'))
    track_df['ts'] = track_df['timestamp']
    track_df['ts_utc'] = pd.to_datetime(track_df['ts'])
    track_df.index = track_df['ts_utc']
    spot_names = sorted(track_df['spotterId'].unique())
    # sample a number for testing
    if args.test_spots > 0:
        spot_names = np.random.choice(spot_names, args.test_spots)
    print(spot_names)
    readers = load_environment_data(args.data_dir, start_time, use_gfs=args.use_gfs, use_ncep=args.use_ncep, use_ww3=args.use_ww3, use_rtofs=args.use_rtofs)
    for spot in spot_names:
        if not os.path.exists(os.path.join(spot_dir, spot + '.nc')):
            simulate_spot(spot, start_datetime=start_time,  end_datetime=end_time,\
                          start_at_drifter=args.start_at_drifter,  end_at_drifter=args.end_at_drifter, \
                          plot_plot=args.plot, plot_gif=args.gif, num_seeds=args.num_seeds,
                          seed_radius=args.seed_radius, wind_drift_factor_max=args.wind_drift_factor_max,
                          model_type=args.model_type, object_type=args.object_type)


