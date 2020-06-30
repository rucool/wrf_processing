#!/usr/bin/env python

"""
Author: Lori Garzio on 6/18/2020
Last modified: 6/18/2020
Creates box plots of wind speeds at each hour of the day from June 2019 - May 2020 at user-defined heights at two
locations: 1) NYSERDA North LiDAR buoy and 2) NYSERDA South LiDAR buoy. The box limits extend from the lower to upper
quartiles, with a line at the median and a diamond symbol at the mean. Whiskers extend from the box to show the
range of the data. Circles indicate outliers.

wrf_dir: path to directory containing RU-WRF 4.1 processed files
sDir: path to save file directory
division: desired method of dividing timestamps, e.g. ['year', 'quarter']
wsheights: list of wind speed heights to plot, e.g. [10, 160]
"""

import numpy as np
import os
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


def append_model_data(nc_filepath, buoy_locs, data_dict, ht):
    """
    Append model data from a specific lat/lon and height to data dictionary
    buoy_name: buoy key name is data_dict
    nc_filepath: file path to NetCDF file containing data
    buoy_locs: dictionary containing buoy latitudes and longitudes
    data_dict: dictionary with keys 't', 'u', 'v' and 'ws' to which data are appended
    ht: height
    """
    ncfile = xr.open_dataset(nc_filepath, mask_and_scale=False)

    lats = ncfile['XLAT']
    lons = ncfile['XLONG']

    for nb, bloc, in buoy_locs.items():
        # Find the closest model point
        # calculate the sum of the absolute value distance between the model location and buoy location
        a = abs(lats - bloc['lat']) + abs(lons - bloc['lon'])

        # find the indices of the minimum value in the array calculated above
        i, j = np.unravel_index(a.argmin(), a.shape)

        # grab the data at that location/index and height
        if ht == 10:
            u = ncfile.U10[:, i, j]
            v = ncfile.V10[:, i, j]
        else:
            u = ncfile.U.sel(height=ht)[:, i, j]
            v = ncfile.V.sel(height=ht)[:, i, j]

        # calculate wind speed (m/s) from u and v
        ws = wind_uv_to_spd(u, v)

        # calculate wind direction (degrees) from u and v
        wd = wind_uv_to_dir(u, v)

        # append data to dictionary
        hour = int(nc_filepath.split('.nc')[0][-3:])
        data_dict[nb][hour]['t'] = np.append(data_dict[nb][hour]['t'], ncfile['Time'].values)
        data_dict[nb][hour]['u'] = np.append(data_dict[nb][hour]['u'], u.values)
        data_dict[nb][hour]['v'] = np.append(data_dict[nb][hour]['v'], v.values)
        data_dict[nb][hour]['ws'] = np.append(data_dict[nb][hour]['ws'], ws.values)
        data_dict[nb][hour]['wd'] = np.append(data_dict[nb][hour]['wd'], wd.values)


def plot_boxplot(data_dict, plt_ttl1, plt_ttl2, save_file):
    """
    Box plots of wind speed and direction
    data_dict: dictionary containing wind speed and direction data at a specific height and location
    plt_ttl: plot title
    save_filepath: file path to save directory and save filename
    """
    plots = ['ws', 'wd']
    for ps in plots:
        bplot = []
        for k, v in data_dict.items():
            bplot.append(list(v[ps]))

        fig, ax = plt.subplots(figsize=(12, 8))

        # customize the boxplot elements
        medianprops = dict(color='black')
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='black')
        boxprops = dict(facecolor='darkgray')

        box = ax.boxplot(bplot, patch_artist=True, labels=list(data_dict.keys()), showmeans=True,
                         medianprops=medianprops, meanprops=meanpointprops, boxprops=boxprops)

        ax.set_xlabel('Hour of Day (GMT)')

        if ps == 'ws':
            ax.set_ylabel('Wind Speed (m/s)')
            ttl = 'Wind Speed'
            sp = 'windsp'
        else:
            ax.set_ylabel('Wind Direction (degrees)')
            ttl = 'Wind Direction'
            sp = 'winddir'

        plt.title('{} {} {}'.format(plt_ttl1, ttl, plt_ttl2))

        save_filepath = '{}_{}.png'.format(save_file, sp)
        plt.savefig(save_filepath, dpi=200)
        plt.close()


def wind_uv_to_dir(u, v):
    """
    Calculates the wind direction from the u and v wind components
    u: west/east direction (wind from the west is positive, from the east is negative)
    v: south/noth direction (wind from the south is positive, from the north is negative)
    wind_dir: wind direction calculated from the u and v wind components
    """
    pi = 3.141592653589793
    wind_dir = np.mod(270-np.arctan2(v, u)*180/pi, 360)

    return wind_dir


def wind_uv_to_spd(u, v):
    """
    Calculates the wind speed from the u and v wind components
    u: west/east direction (wind from the west is positive, from the east is negative)
    v: south/noth direction (wind from the south is positive, from the north is negative)
    WSPD: wind speed calculated from the u and v wind components
    """
    WSPD = np.sqrt(np.square(u)+np.square(v))

    return WSPD


def main(wrf_rawdir, save_dir, divs, heights):
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(nyserda_north=dict(lon=-72.7173, lat=39.9686),
                         nyserda_south=dict(lon=-73.4295, lat=39.5465))
    hours = np.arange(1, 25, 1)

    for height in heights:
        for div in divs:
            if div == 'quarter':
                start_dates = ['06-01-2019', '09-01-2019', '12-01-2019', '03-01-2020']
                end_dates = ['09-01-2019', '12-01-2019', '03-01-2020', '06-01-2020']
            elif div == 'year':
                start_dates = ['06-01-2019']
                end_dates = ['06-01-2020']
                # start_dates = ['06-01-2019']  ### for testing
                # end_dates = ['06-05-2019']  ### for testing

            for t0_str, t1_str in zip(start_dates, end_dates):
                t0 = dt.datetime.strptime(t0_str, '%m-%d-%Y')
                t1 = dt.datetime.strptime(t1_str, '%m-%d-%Y')

                # initialize empty dictionaries for each buoy location and hour of day
                data = dict()
                for key in list(nyserda_buoys.keys()):
                    data[key] = dict()
                    for hr in hours:
                        data[key][hr] = dict(t=np.array([], dtype='datetime64[ns]'),
                                             u=np.array([], dtype='float32'),
                                             v=np.array([], dtype='float32'),
                                             ws=np.array([], dtype='float32'),
                                             wd=np.array([], dtype='float32'))
                # navigate through directories to access files
                for root, dirs, files in os.walk(wrf_rawdir):
                    for dr in sorted(dirs):
                        if t0 <= dt.datetime.strptime(dr, '%Y%m%d') < t1:
                            print('Appending data from {}'.format(dr))
                            for root2, dirs2, files2 in os.walk(os.path.join(root, dr)):
                                for f in sorted(files2):
                                    # append data for hours 1-24
                                    if f.endswith('.nc') and 0 < int(f.split('.nc')[0][-3:]) < 25:
                                    # if f.endswith('.nc') and 0 < int(f.split('.nc')[0][-3:]) < 6:  ##### for testing
                                        append_model_data(os.path.join(root2, f), nyserda_buoys, data, height)

                # plot data for each NYSERDA buoy location for each year/division
                tm0 = np.min(data[key][1]['t'])
                tm1 = np.max(data[key][1]['t'])
                tm_range_str = '{} to {}'.format(pd.to_datetime(tm0).strftime('%b %d %Y'),
                                                 pd.to_datetime(tm1).strftime('%b %d %Y'))
                sdiv = pd.to_datetime(tm0).strftime('%Y%m')

                for loc, da in data.items():
                    if 'north' in loc:
                        buoy = 'NYSERDA North'
                        buoy_code = 'NYNE05'
                    elif 'south' in loc:
                        buoy = 'NYSERDA South'
                        buoy_code = 'NYSE06'
                    ttl1 = 'RU-WRF 4.1: {}m'.format(str(height))
                    ttl2 = 'at {}\n{}'.format(buoy, tm_range_str)

                    # plot box plot
                    sf = 'WRF_boxplot_{}m_{}_{}_{}'.format(height, buoy_code, sdiv, div)
                    sfpath = os.path.join(save_dir, sf)
                    plot_boxplot(da, ttl1, ttl2, sfpath)


if __name__ == '__main__':
    wrf_dir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km'  # on server
    #wrf_dir = '/Volumes/boardwalk/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km'
    #wrf_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/website_plots_redo/processed/3km'
    sDir = '/home/lgarzio/rucool/bpu/wrf/boxplot'  # on server
    #sDir = '/Users/lgarzio/Documents/rucool/bpu/wrf/boxplot'
    division = ['year']  # ['year', 'quarter']
    wsheights = [160]  # [10, 160]
    main(wrf_dir, sDir, division, wsheights)
