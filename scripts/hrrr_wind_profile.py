#!/usr/bin/env python

"""
Author: Lori Garzio on 7/8/2020
Last modified: 7/8/2020
Creates daily profile plots of wind speed from the NOAA HRRR model for hours 0-23 at 2 locations:
1) NYSERDA North LiDAR Buoy
2) NYSERDA South LiDAR Buoy
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})  # set the font size for all plots


def append_hrrr_data(nc_filepath, buoy_locations, data_dict):
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
    a = abs(lats - buoy_locations['lat']) + abs(lons - buoy_locations['lon'])

    # find the indices of the minimum value in the array calculated above
    i, j = np.unravel_index(a.argmin(), a.shape)

    # grab the data at that location/index
    height = ncfile['lv_HTGL1'].values

    for t in ncfile['time'].values:
        windspeeds = []
        hts = []
        for h in height:
            subnc = ncfile.sel(time=t, lv_HTGL1=h)
            windspeeds.append(subnc['wind_speed'].values[i, j])
            hts.append(h)

        # append data to array
        data_dict['t'] = np.append(data_dict['t'], t)
        if len(data_dict['height']) > 0:
            data_dict['height'] = np.vstack((data_dict['height'], hts))
            data_dict['ws'] = np.vstack((data_dict['ws'], windspeeds))
        else:
            data_dict['height'] = hts
            data_dict['ws'] = windspeeds


def plot_wndsp_profile(data_dict, plt_ttl, save_filepath, hmax=None):
    """
    Profile plots of wind speeds, colored by time
    data_dict: dictionary containing wind speed data at multiple heights
    plt_ttl: plot title
    save_filepath: full file path to save directory and save filename
    hmax: optional, maximum height to plot
    """
    n = len(data_dict['t'])
    colors = plt.cm.rainbow(np.linspace(0, 1, n))

    # specify the colorbar tick labels
    hr0 = pd.to_datetime(np.min(data_dict['t'])).hour
    hr1 = pd.to_datetime(np.max(data_dict['t'])).hour
    cbar_labs = np.linspace(hr0, hr1 + 1, 5)
    cbar_labs = ['{:02d}:00'.format(int(x)) for x in cbar_labs]

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(right=0.88, left=0.15)
    plt.grid()
    for i in range(n):
        if hmax is not None:
            height_ind = np.where(data_dict['height'][i] <= hmax)
            ax.plot(data_dict['ws'][i][height_ind], data_dict['height'][i][height_ind], c=colors[i])
        else:
            ax.plot(data_dict['ws'][i], data_dict['height'][i], c=colors[i])
        if i == (n - 1):
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'),
                                ax=ax, orientation='vertical', fraction=0.09, pad=0.03, label='Model Forecast Hour (GMT)')
            cbar.set_ticks([0, .25, .5, .75, 1])
            cbar.ax.set_yticklabels(cbar_labs)
            ax.set_xlabel('Wind Speed (m/s)')
            ax.set_ylabel('Height (m)')
            ax.set_title(plt_ttl)
            if hmax is not None:
                ax.set_xlim(0, 30)
            else:
                ax.set_xlim(0, 40)

            plt.savefig(save_filepath, dpi=200)
            plt.close()


def main(hrrr_rawdir, sdate, edate):
    save_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/hrrr'
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(nyserda_north=dict(lon=-72.7173, lat=39.9686),
                         nyserda_south=dict(lon=-73.4295, lat=39.5465))

    time_span = pd.date_range(sdate, edate, freq='D')

    for ts in time_span:
        f = os.path.join(hrrr_rawdir, 'hrrr_data_{}.nc'.format(ts.strftime('%Y%m%d')))
        data = dict(nyserda_north=dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([])),
                    nyserda_south=dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([])))
        for nb, bloc, in nyserda_buoys.items():
            if 'north' in nb:
                append_hrrr_data(f, bloc, data['nyserda_north'])
            else:
                append_hrrr_data(f, bloc, data['nyserda_south'])

        # plot data for each NYSERDA buoy location
        plt_dt = pd.to_datetime(data['nyserda_north']['t'][0]).strftime('%Y%m%d')
        plt_dt2 = pd.to_datetime(data['nyserda_north']['t'][0]).strftime('%Y-%m-%d')

        hr0 = pd.to_datetime(np.min(data['nyserda_north']['t'])).hour
        hr1 = pd.to_datetime(np.max(data['nyserda_north']['t'])).hour

        for loc, d in data.items():
            if 'north' in loc:
                buoy = 'NYSERDA North'
                buoy_code = 'NYNE05'
            elif 'south' in loc:
                buoy = 'NYSERDA South'
                buoy_code = 'NYSE06'
            ttl = 'RU-WRF 4.1 Wind Profiles at {}\n{}: Hours {:03d}-{:03d}'.format(buoy, plt_dt2, hr0, hr1)

            # plot entire profile
            sf = 'HRRR_wsprofiles_{}_{}_H{:03d}-{:03d}.png'.format(buoy_code, plt_dt, hr0, hr1)
            sfpath = os.path.join(save_dir, sf)
            plot_wndsp_profile(d, ttl, sfpath)


if __name__ == '__main__':
    hrrr_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/hrrr/data/'  # on local machine
    start_date = dt.datetime(2020, 5, 28)
    end_date = dt.datetime(2020, 6, 7)
    main(hrrr_dir, start_date, end_date)
