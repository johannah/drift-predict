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
# from opendrift.readers.reader_grib2 import Reader as Grib2Reader
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

def evaluate_spot(spot_df, pred_nc, results_dict):
    spot = np.array(spot_df['spotterId'])[0]
    # create random wind drift factors
    samples = spot_df.index
    timestamps = spot_df['ts_utc'] 
    # Calculate distances (meters) between simulation and synthetic drifter at each time step
    all_pred_times = np.array(pd.to_datetime(pred_nc['time'][:], utc=True, unit='s'))
    pred_times = []
    pred_lons = []
    pred_lats = []
    use_pred = []
    drifter_lons = []
    drifter_lats = []
    drifter_times = []
    gt_drifter_lons = []
    gt_drifter_lats = []
    gt_drifter_times = []


    eval_time = comp_eval_time
    while eval_time < comp_end_time:
        pred_nearest = np.argmin(abs(all_pred_times-eval_time))
        pred_diff = all_pred_times[pred_nearest] - eval_time

        drift_nearest_idx = spot_df.index.get_loc(eval_time, method='nearest')
        drift_nearest = spot_df.iloc[drift_nearest_idx]
        nearest_ts = drift_nearest['ts_utc']
        drift_diff = nearest_ts - eval_time

        if (abs(pred_diff) < datetime.timedelta(hours=1)) and (abs(drift_diff) < datetime.timedelta(hours=1)):
            gt_drifter_times.append(drift_nearest['ts_utc'])
            gt_drifter_lons.append(drift_nearest['longitude'])
            gt_drifter_lats.append(drift_nearest['latitude'])

            # interpolate bt drifter points
            # drifter sample is older than requested timestep
            if drift_diff < datetime.timedelta(seconds=1):
                prev_idx = drift_nearest_idx
                next_idx = min([drift_nearest_idx+1, spot_df.shape[0]-1])
            else:
                prev_idx = max([drift_nearest_idx-1, 0])
                next_idx = drift_nearest_idx

            prev_pos = (spot_df.iloc[prev_idx]['latitude'], spot_df.iloc[prev_idx]['longitude'])
            next_pos = (spot_df.iloc[next_idx]['latitude'], spot_df.iloc[next_idx]['longitude'])
            diff_dis = haversine(prev_pos, next_pos, unit=Unit.KILOMETERS)
            diff_rad = haversine(prev_pos, next_pos, unit=Unit.RADIANS)
            diff_time_prev_next = spot_df.iloc[next_idx]['ts_utc']-spot_df.iloc[prev_idx]['ts_utc']
            diff_time_prev_target = eval_time-spot_df.iloc[prev_idx]['ts_utc']
            if diff_time_prev_next.seconds > 0:
                diff_time = (diff_time_prev_target.seconds/diff_time_prev_next.seconds)
                interp_dis = diff_time * diff_dis
                if interp_dis > 10:
                    print('interpolating', interp_dis)
                interp_lat, interp_lon = inverse_haversine(prev_pos, diff_dis, diff_rad)
                drifter_lats.append(interp_lat)
                drifter_lons.append(interp_lon)
                drifter_times.append(eval_time)
            else:
                print('no diff bt prev and next', prev_idx, next_idx, eval_time)
                drifter_lats.append(spot_df.iloc[drift_nearest_idx]['latitude']) 
                drifter_lons.append(spot_df.iloc[drift_nearest_idx]['longitude']) 
                drifter_times.append(spot_df.iloc[drift_nearest_idx]['ts_utc'])
 
            pred_times.append(all_pred_times[pred_nearest])
            pred_lons.append(pred_nc['lon'][:][:,pred_nearest])
            pred_lats.append(pred_nc['lat'][:][:,pred_nearest])

        eval_time = eval_time + datetime.timedelta(hours=1) 
    pred_lons = np.array(pred_lons)
    pred_lats = np.array(pred_lats)
    drifter_lons = np.array(drifter_lons)
    drifter_lats = np.array(drifter_lats)
    distances = []
    # plot all predictions
    plt.figure()
    for s in range(pred_lons.shape[1]):
        plt.scatter([pred_lons[0,s]], [pred_lats[0,s]], c='green', marker='o', s=2)
        plt.scatter(pred_lons[:,s], pred_lats[:,s], c='gray', s=2)
        distances.append(distance_between_trajectories(pred_lons[:,s], pred_lats[:,s], drifter_lons, drifter_lats))
    error = np.array(distances).sum(1)
    best_seed = np.argmin(error)
    print('best seed', best_seed, np.max(distances[best_seed]))
    plt.title('best seed %s err %s'%(best_seed, np.max(distances[best_seed])))
    plt.scatter([pred_lons[:,best_seed]], [pred_lats[:,best_seed]], c='c', marker='.', s=2)
    plt.scatter(drifter_lons, drifter_lats, c='b', s=5)
    plt.savefig(os.path.join(load_dir, spot+'_choose.png'))
    plt.close()

    # todo: fix for final submission, check heading names, precision and spotter name format, lat format
    results_dict['spot_id'].append(spot)
    results_dict['text'].append(spot)

    results_dict['lat_day0'].append(drifter_lats[0])  # 12pm EST
    results_dict['lon_day0'].append(drifter_lons[0])  # 12pm EST
    results_dict['pred_lat_day0'].append(pred_lats[0, best_seed])  # 12pm EST
    results_dict['pred_lon_day0'].append(pred_lons[0, best_seed])  # 12pm EST

    results_dict['lat_day2'].append(drifter_lats[48])  # 12pm EST
    results_dict['lon_day2'].append(drifter_lons[48])  # 12pm EST
    results_dict['pred_lat_day2'].append(pred_lats[48, best_seed])  # 12pm EST
    results_dict['pred_lon_day2'].append(pred_lons[48, best_seed])  # 12pm EST
    results_dict['err_day2'].append(distances[best_seed][48])  # 12pm EST

    results_dict['lat_day4'].append(drifter_lats[95])  # 11am EST
    results_dict['lon_day4'].append(drifter_lons[95])  # 11am EST
    results_dict['pred_lat_day4'].append(pred_lats[95, best_seed])  # 11am EST
    results_dict['pred_lon_day4'].append(pred_lons[95, best_seed])  # 11am EST
    results_dict['err_day4'].append(distances[best_seed][95])  # 11am EST (last available)

    day2_err = results_dict['err_day2'][-1]
    day4_err = results_dict['err_day4'][-1]

    distance_error = np.array([4000, 8000, 16000, 32000, np.iinfo(np.int64).max])
    day2_score_multiplier_by_distance = [5, 2, 1, 0, 0]
    day2_score = day2_score_multiplier_by_distance[np.where(day2_err < distance_error)[0][0]]
    day4_score_multiplier_by_distance = [10, 4, 2, 0, 0]
    day4_score = day4_score_multiplier_by_distance[np.where(day4_err < distance_error)[0][0]]

    # day6_score_multiplier_by_distance = [25, 10, 6, 1, 0]
    # day6_score = day6_score_multiplier_by_distance[np.where(day6_err < distance_error)[0][0]]
    # day8_score_multiplier_by_distance = [50, 20, 10, 2, 0]
    # day8_score = day8_score_multiplier_by_distance[np.where(day8_err < distance_error)[0][0]]
    # day10_score_multiplier_by_distance = [100, 40, 20, 5, 0]
    # day10_score = day10_score_multiplier_by_distance[np.where(day10_err < distance_error)[0][0]]

    score = day2_score + day4_score
    results_dict['score'].append(score)

    return np.array(distances), results_dict

def create_results_dict():
    results_dict = {}
    results_dict['spot_id'] = []
    results_dict['text'] = []
    results_dict['lat_day0'] = []
    results_dict['lon_day0'] = []
    results_dict['pred_lat_day0'] = []
    results_dict['pred_lon_day0'] = []
    results_dict['lat_day2'] = []
    results_dict['lon_day2'] = []
    results_dict['pred_lat_day2'] = []
    results_dict['pred_lon_day2'] = []
    results_dict['err_day2'] = []
    results_dict['lat_day4'] = []
    results_dict['lon_day4'] = []
    results_dict['pred_lat_day4'] = []
    results_dict['pred_lon_day4'] = []
    results_dict['err_day4'] = []
    results_dict['score'] = []
    return results_dict

def plot_map(results_dict):
    import pandas as pd
    import plotly.express as px

    # df = pd.DataFrame(dict(lat=[24, 22], lon=[60, 92], subreg=[62, 93]))
    df = pd.DataFrame(results_dict)
    # fig = px.scatter_geo(df, lat="pred_lat_day2", lon="pred_lon_day2", color="err_day2", text="text", range_color=[0, 32000])
    fig = px.scatter_geo(df, lat="pred_lat_day4", lon="pred_lon_day4", color="err_day4", text="text", range_color=[0, 32000])
    fig.update_traces(marker=dict(size=25))
    fig.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('load_dir')
    parser.add_argument('--seed', default=1110)
    parser.add_argument('--eval-type', '-et', default='median', choices=['median'], help='how to choose the lat/lon prediction')
    parser.add_argument('--plot-only', action='store_true', default=False, help='Show error heatmap over map')
    args = parser.parse_args()
    np.random.seed(args.seed)
    load_dir = args.load_dir
    pred_results = glob(os.path.join(args.load_dir, 'SPOT*.nc'))
    print('found %s predictions'%len(pred_results))

    results_dict = create_results_dict()

    if len(pred_results) and not args.plot_only:
        load_args = pickle.load(open(os.path.join(args.load_dir, 'args.pkl'), 'rb'))
        start_time, start_str, end_time, end_str = make_datetimes_from_args(load_args)
        #readers = load_environment(start_time, download=False, use_gfs=load_args.use_gfs, use_ncep=load_args.use_ncep, use_ww3=load_args.use_ww3, use_rtofs=load_args.use_rtofs)
        track_df, wave_df = load_drifter_data(search_path='data/challenge*day*JSON.json', start_date=start_time)
        spot_distances = {}
        running_sum = []
        i = 0
        max_spots = 5  # for testing make this small
        for spot_pred_path in pred_results:
            if i < max_spots:
                spot = (os.path.split(spot_pred_path)[1]).split('.')[0]
                spot_df = track_df[track_df['spotterId'] == spot]
                spot_nc = nc.Dataset(spot_pred_path)
                distance, results_dict = evaluate_spot(spot_df, spot_nc, results_dict)
                min_error = distance.sum(1).min()
                spot_distances[spot] = distance
                print(spot, min_error)
                running_sum.append(min_error)
                i += 1

        print(np.sum(running_sum))
        pickle.dump(spot_distances, open(os.path.join(args.load_dir, 'result_distances.pkl'), 'wb'))

        # todo: fix headings and columns for final submission
        results_df = pd.DataFrame(dict(spotterId=results_dict['spot_id'],
                                       pred_lat_day2=results_dict['pred_lat_day2'], pred_lon_day2=results_dict['pred_lon_day2'],
                                       pred_lat_day4=results_dict['pred_lat_day4'], pred_lon_day4=results_dict['pred_lon_day4'],
                                       score=results_dict['score']))
        results_df.to_csv(os.path.join(args.load_dir, 'results_dict.csv'), sep=',', index=False)

    results_dict = pickle.load(open(os.path.join(args.load_dir, 'error_dict.pkl'), 'rb'))
    plot_map(results_dict)


