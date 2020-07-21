#!/usr/bin/env python

"""
Author: Lori Garzio on 7/7/2020
Last modified: 7/21/2020
Compare WRF 4.1 and NYSERDA wind speed profiles. NYSERDA data from
https://oswbuoysny.resourcepanorama.dnvgl.com/download/f67d14ad-07ab-4652-16d2-08d71f257da1
The WRF 4.1 data point at 158m is interpolated.
"""

import datetime as dt
import numpy as np
import os
import xarray as xr
import pandas as pd
from wrf import interplevel, default_fill
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
plt.rcParams.update({'font.size': 13})  # set the font size for all plots


def append_model_data(nc_filepath, buoy_lat, buoy_lon, data_dict):
    """
    Append model data from a specific lat/lon to data dictionary
    nc_filepath: file path to NetCDF file containing data
    buoy_locations: dictionary containing buoy latitude and longitude
    data_dict: dictionary with keys 't', 'height', and 'ws' to which data are appended
    """
    ncfile = xr.open_dataset(nc_filepath, mask_and_scale=False)

    # interpolate to 158m
    ws_temp = wind_uv_to_spd(ncfile['u'], ncfile['v'])
    ws158 = interplevel(ws_temp, ncfile['height_agl'], 158, default_fill(np.float32))

    lats = ncfile['XLAT']
    lons = ncfile['XLONG']

    # Find the closest model point
    # calculate the sum of the absolute value distance between the model location and buoy location
    a = abs(lats - buoy_lat) + abs(lons - buoy_lon)

    # find the indices of the minimum value in the array calculated above
    i, j = np.unravel_index(a.argmin(), a.shape)

    # grab the data at that location/index
    height = np.squeeze(ncfile['height_agl'])[:, i, j]
    u = np.squeeze(ncfile['u'])[:, i, j]
    v = np.squeeze(ncfile['v'])[:, i, j]

    # calculate wind speed (m/s) from u and v
    ws = wind_uv_to_spd(u, v)

    # find height just below 158m
    max_height = 158 + np.min([n for n in (height.values - 158) if n > 0])
    hind = np.logical_and(height > 50, height < max_height)

    # add the interpolated wind speeds at 158 to the arrays
    heights = np.append(height.values[hind], 158)
    windspeeds = np.append(ws.values[hind], np.squeeze(ws158)[i, j].values)

    # append data to array
    data_dict['t'] = np.append(data_dict['t'], ncfile['Time'].values)
    if len(data_dict['height']) > 0:
        data_dict['height'] = np.vstack((data_dict['height'], heights))
        data_dict['ws'] = np.vstack((data_dict['ws'], windspeeds))
    else:
        data_dict['height'] = heights
        data_dict['ws'] = windspeeds


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    u = west/east direction (wind from the west is positive, from the east is negative)
    v = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WSPD = np.sqrt(np.square(u)+np.square(v))
    return WSPD


def main(wrf_rawdir, nys_dir, sdate, edate, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(NYNE05=dict(lon=-72.7173, lat=39.9686, name='NYSERDA North',
                                     fname='E05_Hudson_North_10_min_avg.csv'),
                         NYSE06=dict(lon=-73.4295, lat=39.5465, name='NYSERDA South',
                                     fname='E06_Hudson_South_10_min_avg.csv'))

    dates = pd.date_range(sdate, edate, freq='D')

    for buoy, value in nyserda_buoys.items():
        print(buoy)
        # get NYSERDA LiDAR data
        nys_ds = pd.read_csv(os.path.join(nys_dir, nyserda_buoys[buoy]['fname']), error_bad_lines=False,
                             delimiter=', ', engine='python')
        cols = [x for x in nys_ds.columns if '_HorizWS' in x]
        nys_ds['timestamp'] = pd.to_datetime(nys_ds['timestamp'])

        for date in dates:
            time_range = pd.date_range(date + dt.timedelta(hours=1), date + dt.timedelta(days=1), freq='H')

            # initialize empty data dictionary
            nys_data = dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([]))

            # for each hour in the date, calculate hourly averages for minutes 10-60 and append to data dictionary
            # for heights 50m - 160m (equals 58m - 158m for LiDAR data)
            for tr in time_range:
                heights = []
                windspeeds = []
                ds = nys_ds[(tr - dt.timedelta(minutes=50) <= nys_ds['timestamp']) & (nys_ds['timestamp'] <= tr)]
                for col in cols:
                    ht = int(col.split('_')[1].split('lidar')[-1].split('m')[0])
                    if 50 <= ht < 160:
                        heights.append(ht)
                        meanws = np.nanmean(pd.to_numeric(ds[col], errors='coerce'))
                        windspeeds.append(meanws)

                # append data to dictionary only if there are no NaNs in the dataset
                if np.sum(np.isnan(windspeeds)) == 0:
                    nys_data['t'] = np.append(nys_data['t'], tr)
                    if len(nys_data['height']) > 0:
                        nys_data['ws'] = np.vstack((nys_data['ws'], windspeeds))
                    else:
                        nys_data['height'] = heights
                        nys_data['ws'] = windspeeds

            # get the WRF 4.1 data for each NYSERDA buoy timestamp with a full profile (between 40-160m height)
            for i, nyst in enumerate(nys_data['t']):
                if nyst.hour == 0:
                    nyst_hr = 24
                    nyst_daystr = (nyst - dt.timedelta(days=1)).strftime('%Y%m%d')
                else:
                    nyst_hr = nyst.hour
                    nyst_daystr = nyst.strftime('%Y%m%d')
                wrfpath = os.path.join(wrf_rawdir, nyst_daystr,
                                       'wrflevs_3km_{}_00Z_H{:03d}.nc'.format(nyst_daystr, nyst_hr))
                wrfdata = dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([]))
                append_model_data(wrfpath, value['lat'], value['lon'], wrfdata)

                nys_height = np.array(nys_data['height'])
                if len(nys_data['t']) > 1:
                    nys_ws = nys_data['ws'][i]
                else:
                    nys_ws = nys_data['ws']
                wrf_height = wrfdata['height']
                wrf_ws = wrfdata['ws']

                # append data points to create a polygon
                polygon_points = []

                # append all xy points for curve 1
                for j, k in enumerate(nys_ws):
                    polygon_points.append([k, nys_height[j]])

                # append all xy points for curve 2 in the reverse order (from last point to first point)
                r_wrf_height = wrf_height[::-1]
                r_wrf_ws = wrf_ws[::-1]
                for j, k in enumerate(r_wrf_ws):
                    polygon_points.append([k, r_wrf_height[j]])

                # append the first point in curve 1 again, so it closes the polygon
                polygon_points.append([nys_ws[0], nys_height[0]])

                polygon = Polygon(polygon_points)
                area = polygon.area

                ttl = 'RU-WRF 4.1 and {} Wind Profiles\n{} UTC, area = {}'.format(value['name'],
                                                                                  nyst.strftime('%Y-%m-%d %H:%M'),
                                                                                  round(area, 2))

                # plot
                sf = 'WRF_{}_wsprofile_compare_{}_H{:03d}.png'.format(buoy, nyst_daystr, nyst_hr)
                sfpath = os.path.join(save_dir, sf)

                fig, ax = plt.subplots(figsize=(8, 9))
                plt.subplots_adjust(right=0.88, left=0.15)
                plt.grid()
                ax.plot(wrf_ws, wrf_height, marker='.', markersize=10, c='tab:blue', label='WRF 4.1')
                ax.plot(nys_ws, nys_height, marker='.', markersize=10, c='tab:orange', label=value['name'])
                ax.fill(*polygon.exterior.xy, c='gray', alpha=.4)
                ax.set_xlabel('Wind Speed (m/s)')
                ax.set_ylabel('Height (m)')
                ax.set_title(ttl)
                plt.legend()

                plt.savefig(sfpath, dpi=200)
                plt.close()


if __name__ == '__main__':
    # wrfdir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/modlevs/3km' on server
    wrfdir = '/Users/lgarzio/Documents/rucool/bpu/wrf/profile_plots/processed/modlevs/3km'  # on local machine
    nyserda_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/data'
    start_date = dt.datetime(2020, 6, 2)
    end_date = dt.datetime(2020, 6, 3)
    sDir = '/Users/lgarzio/Documents/rucool/bpu/wrf/profile_comparisons'
    main(wrfdir, nyserda_dir, start_date, end_date, sDir)
