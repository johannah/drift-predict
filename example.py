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
px.set_mapbox_access_token('pk.eyJ1IjoiamgxNzM2IiwiYSI6ImNpaG8wZWNnYjBwcGh0dGx6ZG1mMGl0czAifQ.mhmvIGx34x2fw0s3p9pnaw')

from haversine import haversine, Unit, inverse_haversine, Direction
from IPython import embed
# https://polar.ncep.noaa.gov/waves/viewer.shtml?-multi_1-US_eastcoast-
# download_data()

from opendrift.readers.reader_current_from_drifter import Reader as DrifterReader
from opendrift.readers.reader_current_from_track import Reader as TrackReader
from opendrift.readers.reader_netCDF_CF_generic import Reader as GenericReader
from opendrift.readers.reader_netCDF_CF_unstructured import Reader as UnstructuredReader
from opendrift.models.openberg import OpenBerg
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.physics_methods import wind_drift_factor_from_trajectory, distance_between_trajectories
from utils import DATA_DIR, make_weather_clips
#from mpl_toolkits.basemap import Basemap
#import numpy as np
#import matplotlib.pyplot as plt
#from pylab import *
#import netCDF4
# how to load nomads data: 
# https://polar.ncep.noaa.gov/global/examples/usingpython.shtml


# how far do drifters go in 10 days?
# TODO find wind drift factor
wind_drift_factor = 0.033
seed_radius = 10 # in meters
num_seeds = 10
track_df, wave_df = load_data(search_path='data/challenge_*day15JSON.json')

#track_df, wave_df = load_data(search_path='data/challenge_*day*JSON.json')
spot_names = sorted(track_df['spotterId'].unique())[:1]
print(spot_names)
#spot_extent_dict, extent_dict = make_weather_clips(spot_names, track_df)
spot_extent_dict = {spot_names[0]:(43.0,47.0,-44.0,-40.0)}
extent_dict = {(43.0,47.0,-44.0,-40.0):['/Volumes/seahorse/2021-drifters/pred/20211118/clips/rtofs_glo_2ds_f003_prog_clip_43.0_47.0_-44.0_-40.0.nc']}
for spot, extent in spot_extent_dict.items():
    spot_df = track_df[track_df['spotterId'] == spot]
    samples = spot_df.index
    ot = OceanDrift(loglevel=0)
    #ot.add_readers_from_list(extent_dict[extent], lazy=False)
    r = DrifterReader(
        lons=spot_df['longitude'], lats=spot_df['latitude'], times=spot_df['ts_utc'])
    ot.add_reader(r)

    # TODO UTC or East?
    start_sample = samples[0]
    end_sample = samples[-1]
    start_time = spot_df.loc[start_sample, 'ts_utc']
    end_time =   spot_df.loc[end_sample, 'ts_utc']
    start_lon = spot_df.loc[start_sample, 'longitude']
    start_lat = spot_df.loc[start_sample, 'latitude']
    # TODO how to do start time
    #time=reader_current.start_time,
    ot.seed_elements(start_lon, start_lat, radius=seed_radius, number=num_seeds,
                time=start_time,
                wind_drift_factor=wind_drift_factor)

    # time step should be in seconds
    ot.run(end_time=end_time, time_step=r.time_step.seconds)
    ot.animation(buffer=.01, fast=True, drifter={'time': spot_df['ts_utc'], 'lon': spot_df['longitude'], 'lat': spot_df['latitude'],
                                          'label': 'CODE Drifter', 'color': 'b', 'linewidth': 2, 'markersize': 40})
    # Drifter track is shown in red, and simulated trajectories are shown in gray. 
    ot.plot(buffer=.01, fast=True, trajectory_dict={
            'lon':spot_df['longitude'], 'lat':spot_df['latitude'],
            'time':samples, 'linestyle': 'r-'})
embed()

#data = nc.Dataset(load_fpath)
#mydate='20211118';
#baseurl =   '//nomads.ncep.noaa.gov:9090/dods/'+ 'rtofs/rtofs_global'
#product_url =  '/rtofs_glo_3dz_forecast_daily_temp_nopdef'
#
## Note that the NOMADS data server interpolates and delivers the data on a regular 
## lat/lon field, not the native model grid. To analyze the model output on the 
## native grid you will have to work from a downloaded NetCDF file (see Example 2), 
## or use a NOPDEF url from the Developmental NOMADS. The latter are experimental 
## links that bypass the grid interpolation step. 
## For example, the NOPDEF URL would be as follows:
#baseurl2 =  '//nomad1.ncep.noaa.gov:9090/dods/' + 'rtofs_global/rtofs.'
#product_url2 = '/rtofs_glo_3dz_forecast_daily_temp'
#
#url=baseurl+mydate+product_url
#
## Extract the sea surface temperature field from NOMADS.
#file = netCDF4.Dataset(url)
#lat  = file.variables['lat'][:]
#lon  = file.variables['lon'][:]
#data      = file.variables['temperature'][1,1,:,:]
#file.close()
#m=Basemap(projection='mill',lat_ts=10, \
#  llcrnrlon=lon.min(),urcrnrlon=lon.max(), \
#  llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
#  resolution='c')
## Convert the lat/lon values to x/y projections.
#Lon, Lat = meshgrid(lon,lat)
#x, y = m(Lon,Lat)
## Next, plot the field using the fast pcolormesh routine and set the colormap to jet.
#cs = m.pcolormesh(x,y,data,shading='flat', \
#  cmap=plt.cm.jet)
## Add a coastline and axis values.
#m.drawcoastlines()
#m.fillcontinents()
#m.drawmapboundary()
#m.drawparallels(np.arange(-90.,120.,30.), \
#  labels=[1,0,0,0])
#m.drawmeridians(np.arange(-180.,180.,60.), \
#  labels=[0,0,0,1])
## Add a colorbar and title, and then show the plot.
#colorbar(cs)
#plt.title('Example 1: Global RTOFS SST from NOMADS')
#plt.show()
## You should see this image in your figure window: figure for example 1
#
##Example 2: Plot data from an Global RTOFS NetCDF file
## This example requires that you download a NetCDF file from either our NOMADS data server 
## or the Production FTP Server (see our Data Access page for more information. 
## For this exercise, I used the nowcast file for 20111004: rtofs_glo_3dz_n048_daily_3ztio.nc 
## retrieved from NOMADS. This example assumes that the NetCDF file is in the 
## current working directory.
##
### Begin by importing the necessary modules and set up the figure
##
##from mpl_toolkits.basemap import Basemap
##import numpy as np
##import matplotlib.pyplot as plt
##from pylab import *
##import netCDF4
##
##plt.figure()
##nc='rtofs_glo_3dz_n048_daily_3ztio.nc';
##In this example we will extract the surface temperature field from the model. Remember that indexing in Python starts at zero.
##file = netCDF4.Dataset(nc)
##lat  = file.variables['Latitude'][:]
##lon  = file.variables['Longitude'][:]
##data = file.variables['temperature'][0,0,:,:]
##file.close()
##
### There is a quirk to the global NetCDF files that isn't in the NOMADS data, 
### namely that there are junk values of longitude (lon>500) in the rightmost column 
### of the longitude array (they are ignored by the model itself). So we have to work around them a little with NaN substitution.
##lon = np.where(np.greater_equal(lon,500),np.nan,lon)
### From this point on the code is almost identical to the previous example. 
### The missing step is that since lat/lon values are arrays in the native grid instead of vectors returned by NOMADS, the meshgrid step is unnecessary.
### Plot the field using Basemap. Start with setting the map projection using the 
### limits of the lat/lon data itself (note that we're using the lonmin and lonmax values computed previously):
##
##plt.figure()
##m=Basemap(projection='mill',lat_ts=10, \
##  llcrnrlon=np.nanmin(lon),urcrnrlon=np.nanmax(lon), \
##  llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
##  resolution='c')
### Convert the lat/lon values to x/y projections.
##x, y = m(lon,lat)
### Next, plot the field using the fast pcolormesh routine and set the colormap to jet.
##cs = m.pcolormesh(x,y,data,shading='flat', cmap=plt.cm.jet)
### Add a coastline and axis values.
##m.drawcoastlines()
##m.fillcontinents()
##m.drawmapboundary()
##m.drawparallels(np.arange(-90.,120.,30.), \
##  labels=[1,0,0,0])
##m.drawmeridians(np.arange(-180.,180.,60.), \
##  labels=[0,0,0,1])
### Add a colorbar and title, and then show the plot.
##colorbar(cs)
##plt.title('Example 2: Global RTOFS SST from NetCDF')
##plt.show()
###track_df, wave_df = load_data(search_path='data/challenge_*day3JSON.json')
###data = {}
###for sc, spot in enumerate(track_df['spotterId'].unique()):
###    data[spot] = track_df[track_df['spotterId'] == spot]
###    if not data[spot].shape[0]:
###        print('spot not available', spot)
###    else:
###        embed()
###        reader = DrifterReader(lons=data[spot]['longitude'], lats=data[spot]['latitude'], 
###                  times=data[spot]['ts_east'])
###
##
