import matplotlib
matplotlib.use("Agg")
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

def make_dataset(spot_names, model_names, model_dicts):
    
    train_data = [] 
    for spot in spot_names:
        spot_data = []
        for model in model_names:
            data = model_dicts[model][spot]
def make_summary_file(load_dir, data_start_time, data_seed_time, data_eval_times):
    model_name = os.path.split(load_dir)[1]
    model_dict = {}
    model_dict['start_time'] = data_start_time
    model_dict['seed_time'] = data_seed_time
    model_dict['eval_time'] = data_eval_times

    pred_results = glob(os.path.join(load_dir, 'SPOT*.nc'))
    thru_results = glob(os.path.join(load_dir, 'SPOT*thru_model.pkl'))
    assert len(pred_results) == len(thru_results)
    #assert len(pred_results) == 88
    load_args = pickle.load(open(os.path.join(load_dir, 'args.pkl'), 'rb'))
    model_dict['load_args'] = load_args
    args_start_time, args_start_str, args_end_time, args_end_str = make_datetimes_from_args(load_args)
    for pr in pred_results:
        pred_nc = nc.Dataset(pr)
        thru = pickle.load(open(pr.replace('.nc', '_drifter_thru_model.pkl'), 'rb'))
        spot = os.path.split(pr)[1].replace('.nc', '')
        if spot in ['SPOT-010057', 'SPOT-1166']:
            continue

        pred_time_raw = pred_nc['time'][:].data
        pred_time = np.array([t.replace(tzinfo=None) for t in (pd.to_datetime(pred_nc['time'][:], utc=True, unit='s'))])
        pred_lats = pred_nc['lat'][:].data
        pred_lons = pred_nc['lon'][:].data
        pred_x_wind = pred_nc['x_wind'][:].data
        pred_y_wind = pred_nc['y_wind'][:].data
        pred_x_sea_water_velocity = pred_nc['x_sea_water_velocity'][:].data
        pred_y_sea_water_velocity = pred_nc['y_sea_water_velocity'][:].data

        drift_time = np.array(thru['time'])
        drift_lats = np.array(thru['lat'])
        drift_lons = np.array(thru['lon'])
        drift_x_wind = np.array(thru['x_wind'])
        drift_y_wind = np.array(thru['y_wind'])
        drift_x_sea_water_velocity = np.array(thru['x_sea_water_velocity'])
        drift_y_sea_water_velocity = np.array(thru['y_sea_water_velocity'])


        # priming with drift data
        prime_idxs = np.array([ii for ii, t in enumerate(drift_time) if (data_start_time < t <  data_seed_time)])
        if not len(prime_idxs):
            print("no data available for", spot)
        else:
            priming_times = drift_time[prime_idxs]
            priming = np.vstack((
                                drift_lats[prime_idxs], 
                                drift_lons[prime_idxs], 
                                drift_x_wind[prime_idxs], 
                                drift_y_wind[prime_idxs], 
                                drift_x_sea_water_velocity[prime_idxs], 
                                drift_y_sea_water_velocity[prime_idxs]))

            # condition on available predictions
            # first eval is when we stop conditioning
            cond_idxs = np.array([ii for ii, t in enumerate(pred_time) if (data_seed_time < t <= data_eval_times[0])])
            conditioning_times = pred_time[cond_idxs]
            conditioning = np.vstack((
                pred_lats[:, cond_idxs], 
                pred_lons[:, cond_idxs], 
                pred_x_wind[:, cond_idxs], 
                pred_y_wind[:, cond_idxs], 
                pred_x_sea_water_velocity[:, cond_idxs], 
                pred_y_sea_water_velocity[:, cond_idxs]))


            eval_lats = []
            eval_lons = []
            eval_times = []
            for eval_time in data_eval_times:
                drift_nearest_idx = np.argmin(abs(drift_time-eval_time))
                drift_nearest_ts = drift_time[drift_nearest_idx]
                drift_diff = drift_nearest_ts - eval_time

                if (abs(drift_diff) > datetime.timedelta(hours=3)):
                    print('long time', abs(drift_diff), eval_time)
                    embed()
                # interpolate bt drifter points
                # drifter sample is older than requested timestep
                if drift_diff <= datetime.timedelta(seconds=1):
                    prev_idx = drift_nearest_idx
                    next_idx = min([drift_nearest_idx+1, drift_time.shape[0]-1])
                else:
                    prev_idx = max([drift_nearest_idx-1, 0])
                    next_idx = drift_nearest_idx

                prev_pos = (drift_lats[prev_idx], drift_lons[prev_idx])
                next_pos = (drift_lats[next_idx], drift_lons[next_idx])
                diff_dis = haversine(prev_pos, next_pos, unit=Unit.KILOMETERS)
                diff_rad = haversine(prev_pos, next_pos, unit=Unit.RADIANS)
                diff_time_prev_next = drift_time[next_idx]-drift_time[prev_idx]
                diff_time_prev_target = eval_time-drift_time[prev_idx]
                if diff_time_prev_next.seconds > 0:
                    diff_time = (diff_time_prev_target.seconds/diff_time_prev_next.seconds)
                    interp_dis = diff_time * diff_dis
                    if interp_dis > 10:
                        print('interpolating', interp_dis)
                        embed()
                    target_lat, target_lon = inverse_haversine(prev_pos, diff_dis, diff_rad)
                else:
                    target_lat = drift_lats[drift_nearest_idx]
                    target_lon = drift_lons[drift_nearest_idx]
                eval_times.append(drift_nearest_ts)
                eval_lats.append(target_lat)
                eval_lons.append(target_lon)
 
            model_dict[spot] = {}
            model_dict[spot]['priming'] = priming
            model_dict[spot]['priming_times'] = priming_times
            model_dict[spot]['conditioning'] = conditioning
            model_dict[spot]['conditioning_times'] = conditioning_times
            model_dict[spot]['target_lats'] = eval_lats 
            model_dict[spot]['target_lons'] = eval_lons
            model_dict[spot]['target_times'] = eval_times
            print(conditioning.shape)
            print(priming.shape)

    pickle.dump(model_dict, open(os.path.join(load_dir, 'summary.pkl'), 'wb'))
#        #pred_idxs = []
#        #drifter_lons = []
#        #drifter_lats = []
#        #drifter_times = []
#        #gt_drifter_lons = []
#        #gt_drifter_lats = []
#        #gt_drifter_times = []
#
#
#    eval_time = comp_eval_time
#    while eval_time < comp_end_time:
#        pred_nearest = np.argmin(abs(all_pred_times-eval_time))
#        pred_diff = all_pred_times[pred_nearest] - eval_time
#
#        drift_nearest_idx = spot_df.index.get_loc(eval_time, method='nearest')
#        drift_nearest = spot_df.iloc[drift_nearest_idx]
#        nearest_ts = drift_nearest['ts_utc']
#        drift_diff = nearest_ts - eval_time
#
#        if (abs(pred_diff) < datetime.timedelta(hours=1)) and (abs(drift_diff) < datetime.timedelta(hours=1)):
#            gt_drifter_times.append(drift_nearest['ts_utc'])
#            gt_drifter_lons.append(drift_nearest['longitude'])
#            gt_drifter_lats.append(drift_nearest['latitude'])
#            pred_idxs.append(pred_nearest)
#
#            # interpolate bt drifter points
#            # drifter sample is older than requested timestep
#            if drift_diff < datetime.timedelta(seconds=1):
#                prev_idx = drift_nearest_idx
#                next_idx = min([drift_nearest_idx+1, spot_df.shape[0]-1])
#            else:
#                prev_idx = max([drift_nearest_idx-1, 0])
#                next_idx = drift_nearest_idx
#
#            prev_pos = (spot_df.iloc[prev_idx]['latitude'], spot_df.iloc[prev_idx]['longitude'])
#            next_pos = (spot_df.iloc[next_idx]['latitude'], spot_df.iloc[next_idx]['longitude'])
#            diff_dis = haversine(prev_pos, next_pos, unit=Unit.KILOMETERS)
#            diff_rad = haversine(prev_pos, next_pos, unit=Unit.RADIANS)
#            diff_time_prev_next = spot_df.iloc[next_idx]['ts_utc']-spot_df.iloc[prev_idx]['ts_utc']
#            diff_time_prev_target = eval_time-spot_df.iloc[prev_idx]['ts_utc']
#            if diff_time_prev_next.seconds > 0:
#                diff_time = (diff_time_prev_target.seconds/diff_time_prev_next.seconds)
#                interp_dis = diff_time * diff_dis
#                if interp_dis > 10:
#                    print('interpolating', interp_dis)
#                interp_lat, interp_lon = inverse_haversine(prev_pos, diff_dis, diff_rad)
#                drifter_lats.append(interp_lat)
#                drifter_lons.append(interp_lon)
#                drifter_times.append(eval_time)
#            else:
#                print('no diff bt prev and next', prev_idx, next_idx, eval_time)
#                drifter_lats.append(spot_df.iloc[drift_nearest_idx]['latitude']) 
#                drifter_lons.append(spot_df.iloc[drift_nearest_idx]['longitude']) 
#                drifter_times.append(spot_df.iloc[drift_nearest_idx]['ts_utc'])
# 
#
#        eval_time = eval_time + datetime.timedelta(hours=1) 
#        #eval_time = eval_time + datetime.timedelta(days=1) 
#    drifter_lons = np.array(drifter_lons)
#    drifter_lats = np.array(drifter_lats)
#    embed()
#    # plot all predictions
#    #plt.figure()
#    #for s in range(pred_lons.shape[1]):
#    #    plt.scatter([pred_lons[0,s]], [pred_lats[0,s]], c='green', marker='o', s=2)
#    #    plt.scatter(pred_lons[:,s], pred_lats[:,s], c='gray', s=2)
#    #    distances.append(distance_between_trajectories(pred_lons[:,s], pred_lats[:,s], drifter_lons, drifter_lats))
#    #error = np.array(distances).sum(1)
#    #best_seed = np.argmin(error)
#    #print('best seed', best_seed, np.max(distances[best_seed]))
#    #plt.title('best seed %s err %s'%(best_seed, np.max(distances[best_seed])))
#    #plt.scatter([pred_lons[:,best_seed]], [pred_lats[:,best_seed]], c='c', marker='.', s=2)
#    #plt.scatter(drifter_lons, drifter_lats, c='b', s=5)
#    #plt.savefig(os.path.join(load_dir, spot+'_choose.png'))
#    #plt.close()
#    return np.array(distances)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1110)
    parser.add_argument('--eval-type', '-et', default='median', choices=['median'], help='how to choose the lat/lon prediction')
    args = parser.parse_args()
    np.random.seed(args.seed)
    #start_time = datetime.datetime(2021, 11, 17, 17, 0, tzinfo=pytz.UTC)
    #seed_time = datetime.datetime(2021, 11, 19, 17, 0, tzinfo=pytz.UTC)
    #eval_time = datetime.datetime(2021, 11, 21, 17, 0, tzinfo=pytz.UTC)


    # FOR TRAINING
    start_time = datetime.datetime(2021, 11, 17, 17, 0, tzinfo=None)
    seed_time = datetime.datetime(2021, 11, 19, 17, 0, tzinfo=None)
    eval_times = [datetime.datetime(2021, 11, 20, 17, 0, tzinfo=None),
                  datetime.datetime(2021, 11, 21, 17, 0, tzinfo=None),
                  datetime.datetime(2021, 11, 22, 17, 0, tzinfo=None)]

    load_data_from_dirs = [
            '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1945_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway71',
            '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway69', 
            '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway72', 
            '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_WD0.06_OceanDrift', ]


    ## FOR EVALUATION - order of dirs is important
    #start_time = datetime.datetime(2021, 11, 20, 17, 0, tzinfo=None)
    #seed_time = datetime.datetime(2021, 11, 22, 17, 0, tzinfo=None)
    #eval_times = [datetime.datetime(2021, 11, 24, 17, 0, tzinfo=None),
    #              datetime.datetime(2021, 11, 26, 17, 0, tzinfo=None),
    #              datetime.datetime(2021, 11, 28, 17, 0, tzinfo=None)
    #              datetime.datetime(2021, 11, 30, 17, 0, tzinfo=None)
    #              datetime.datetime(2021, 12, 2, 17, 0, tzinfo=None)
    #              ]

    #load_data_from_dirs = [
    #        '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1945_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway71',
    #        '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway69', 
    #        '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_Leeway72', 
    #        '/Volumes/seahorse/2021-drifters/results/spots_N20211122-1947_S20211117-1700_E20211203-1700_DS0_DE1_R1G1W1N0_WD0.06_OceanDrift', ]



    for d in load_data_from_dirs:
        summary_file = os.path.join(d, 'summary.pkl') 
        if not os.path.exists(summary_file):
            make_summary_file(d, start_time, seed_time, eval_times)

#    print('found %s predictions'%len(pred_results))
#    if len(pred_results):
#        load_args = pickle.load(open(os.path.join(args.load_dir, 'args.pkl'), 'rb'))
#        start_time, start_str, end_time, end_str = make_datetimes_from_args(load_args)
#        track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json', start_date=start_time)

