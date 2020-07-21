#!/usr/bin/env python

"""
Author: Lori Garzio on 7/8/2020
Last modified: 7/8/2020
Compare NOAA HRRR and NYSERDA wind speed profiles. NYSERDA data from
https://oswbuoysny.resourcepanorama.dnvgl.com/download/f67d14ad-07ab-4652-16d2-08d71f257da1
"""

import datetime as dt
import numpy as np
import os
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
plt.rcParams.update({'font.size': 13})  # set the font size for all plots


def append_hrrr_data(nc_filepath, modeltm, buoy_lat, buoy_lon, data_dict):
    """
    Append model data from a specific lat/lon to data dictionary
    nc_filepath: file path to NetCDF file containing data
    buoy_locations: dictionary containing buoy latitude and longitude
    data_dict: dictionary with keys 't', 'height', and 'ws' to which data are appended
    """
    ncfile = xr.open_dataset(nc_filepath, mask_and_scale=False)

    lats = ncfile['gridlat_0']
    lons = ncfile['gridlon_0']

    # Find the closest model point
    # calculate the sum of the absolute value distance between the model location and buoy location
    a = abs(lats - buoy_lat) + abs(lons - buoy_lon)

    # find the indices of the minimum value in the array calculated above
    i, j = np.unravel_index(a.argmin(), a.shape)

    # grab the data at that location/index
    height = ncfile['lv_HTGL1'].values
    windspeeds = []
    hts = []
    for h in height:
        subnc = ncfile.sel(time=modeltm, lv_HTGL1=h)
        windspeeds.append(subnc['wind_speed'].values[i, j])
        hts.append(h)

    # append data to array
    data_dict['t'] = np.append(data_dict['t'], subnc['time'].values)
    if len(data_dict['height']) > 0:
        data_dict['height'] = np.vstack((data_dict['height'], hts))
        data_dict['ws'] = np.vstack((data_dict['ws'], windspeeds))
    else:
        data_dict['height'] = hts
        data_dict['ws'] = windspeeds


def main(hrrr_rawdir, nys_dir, sdate, edate, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(NYNE05=dict(lon=-72.7173, lat=39.9686, name='NYSERDA North',
                                     fname='E05_Hudson_North_10_min_avg-test.csv'),
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
            # for heights 10m - 90m
            for tr in time_range:
                heights = []
                windspeeds = []
                ds = nys_ds[(tr - dt.timedelta(minutes=50) <= nys_ds['timestamp']) & (nys_ds['timestamp'] <= tr)]
                for col in cols:
                    ht = int(col.split('_')[1].split('lidar')[-1].split('m')[0])
                    if 10 <= ht < 90:
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

            # get the HRRR data for each NYSERDA buoy timestamp with a full profile (between 10-90m height)
            for i, nyst in enumerate(nys_data['t']):
                nyst_hr = nyst.hour
                nyst_daystr = nyst.strftime('%Y%m%d')
                hrrr_path = os.path.join(hrrr_rawdir, 'hrrr_data_{}.nc'.format(nyst_daystr))
                hrrr_data = dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([]))
                append_hrrr_data(hrrr_path, nyst, value['lat'], value['lon'], hrrr_data)

                nys_height = np.array(nys_data['height'])
                if len(nys_data['t']) > 1:
                    nys_ws = nys_data['ws'][i]
                else:
                    nys_ws = nys_data['ws']
                hrrr_height = hrrr_data['height']
                hrrr_ws = hrrr_data['ws']

                # append data points to create a polygon
                polygon_points = []

                # append all xy points for curve 1
                for j, k in enumerate(nys_ws):
                    polygon_points.append([k, nys_height[j]])

                # append all xy points for curve 2 in the reverse order (from last point to first point)
                r_wrf_height = hrrr_height[::-1]
                r_wrf_ws = hrrr_ws[::-1]
                for j, k in enumerate(r_wrf_ws):
                    polygon_points.append([k, r_wrf_height[j]])

                # append the first point in curve 1 again, so it closes the polygon
                polygon_points.append([nys_ws[0], nys_height[0]])

                polygon = Polygon(polygon_points)
                area = polygon.area

                ttl = 'HRRR and {} Wind Profiles\n{} UTC, area = {}'.format(value['name'],
                                                                            nyst.strftime('%Y-%m-%d %H:%M'),
                                                                            round(area, 2))

                # plot
                sf = 'HRRR_{}_wsprofile_compare_{}_H{:03d}.png'.format(buoy, nyst_daystr, nyst_hr)
                sfpath = os.path.join(save_dir, sf)

                fig, ax = plt.subplots(figsize=(8, 9))
                plt.subplots_adjust(right=0.88, left=0.15)
                plt.grid()
                ax.plot(hrrr_ws, hrrr_height, marker='.', markersize=10, c='tab:purple', label='HRRR')
                ax.plot(nys_ws, nys_height, marker='.', markersize=10, c='tab:orange', label=value['name'])
                ax.fill(*polygon.exterior.xy, c='gray', alpha=.4)
                ax.set_xlabel('Wind Speed (m/s)')
                ax.set_ylabel('Height (m)')
                ax.set_title(ttl)
                plt.legend()

                plt.savefig(sfpath, dpi=200)
                plt.close()


if __name__ == '__main__':
    hrrr_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/hrrr/data/'  # on local machine
    nyserda_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/data'
    start_date = dt.datetime(2020, 5, 29)
    end_date = dt.datetime(2020, 5, 29)
    sDir = '/Users/lgarzio/Documents/rucool/bpu/wrf/profile_comparisons'
    main(hrrr_dir, nyserda_dir, start_date, end_date, sDir)
