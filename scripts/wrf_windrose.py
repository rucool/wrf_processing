#!/usr/bin/env python

"""
Author: Lori Garzio on 6/16/2020
Last modified: 6/30/2020
Creates wind rose plots for user-defined time ranges and heights at two locations: 1) NYSERDA North LiDAR buoy and
2) NYSERDA South LiDAR buoy.

wrf_dir: path to directory containing RU-WRF 4.1 processed files
sDir: path to save file directory
years: list of years in string format to plot, e.g. ['2019']
division: desired method of dividing timestamps, e.g. 'monthly'
wsheights: list of wind speed heights to plot, e.g. [10, 160]
"""

import numpy as np
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes


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

        # append data to dictionary
        data_dict[nb]['t'] = np.append(data_dict[nb]['t'], ncfile['Time'].values)
        data_dict[nb]['u'] = np.append(data_dict[nb]['u'], u.values)
        data_dict[nb]['v'] = np.append(data_dict[nb]['v'], v.values)
        data_dict[nb]['ws'] = np.append(data_dict[nb]['ws'], ws.values)


def new_axes():
    """
    Create new wind rose axes
    """
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = WindroseAxes(fig, rect, facecolor='w')
    fig.add_axes(ax)
    return ax


def plot_windrose(data_dict, plt_ttl, save_filepath):
    """
    Wind rose plots of wind speed and direction
    data_dict: dictionary containing wind speed data at a specific height and location
    plt_ttl: plot title
    save_filepath: full file path to save directory and save filename
    """
    # calculate wind direction from u and v
    wd = wind_uv_to_dir(data_dict['u'], data_dict['v'])
    # ccodes = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']  # yellow blue
    # ccodes = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']  # yellow red
    # ccodes = ['#c7e9b4', '#41b6c4', '#756bb1', '#54278f', '#fc8d59', '#b30000']  # blues purples reds

    ax = new_axes()

    # set the bins for wind speeds
    b = [0, 5, 10, 15, 20, 25, 30]
    # ax.bar(wd, data_dict['ws'], normed=True, bins=b, opening=1, edgecolor='black', colors=ccodes, nsector=36)
    # ax.bar(wd, data_dict['ws'], normed=True, opening=1, edgecolor='black', cmap=cm.viridis, nsector=36)
    ax.bar(wd, data_dict['ws'], normed=True, bins=b, opening=1, edgecolor='black', cmap=cm.jet, nsector=36)

    # add % to y-axis labels
    newticks = ['{:.0%}'.format(x/100) for x in ax.get_yticks()]
    ax.set_yticklabels(newticks)

    # format legend
    set_legend(ax)

    # add title
    ax.set_title(plt_ttl, fontsize=14)

    plt.savefig(save_filepath, dpi=200)
    plt.close()


def set_legend(ax):
    """
    Adjust the wind rose legend box
    """
    # move legend
    al = ax.legend(borderaxespad=-7, title='Wind Speed (m/s)')

    # replace the text in the legend
    text_str = ['0$\leq$ ws <5', '5$\leq$ ws <10', '10$\leq$ ws <15', '15$\leq$ ws <20', '20$\leq$ ws <25',
                '25$\leq$ ws <30', 'ws $\geq$30']
    for i, txt in enumerate(al.get_texts()):
        txt.set_text(text_str[i])
    plt.setp(al.get_texts(), fontsize=10)


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


def main(wrf_rawdir, save_dir, yrs, divs, heights):
    os.makedirs(save_dir, exist_ok=True)

    # locations of NYSERDA LIDAR buoys
    nyserda_buoys = dict(nyserda_north=dict(lon=-72.7173, lat=39.9686),
                         nyserda_south=dict(lon=-73.4295, lat=39.5465))

    if divs == 'monthly':
        div = list(range(1, 13))
        #div = [1]  ##### for testing

    for height in heights:
        for yr in yrs:
            for d in div:
                # initialize empty dictionaries for each buoy location
                data = dict()
                for key in list(nyserda_buoys.keys()):
                    data[key] = dict(t=np.array([], dtype='datetime64[ns]'), u=np.array([], dtype='float32'),
                                     v=np.array([], dtype='float32'), ws=np.array([], dtype='float32'))
                # navigate through directories to access files
                for root, dirs, files in os.walk(wrf_rawdir):
                    for dr in sorted(dirs):
                        if dr[0:4] == yr and int(dr[4:6]) == d:
                        # if dr[0:4] == yr and int(dr[4:6]) == d and int(dr[6:8]) < 5:  ##### for testing
                            print('Appending data from {}'.format(dr))
                            for root2, dirs2, files2 in os.walk(os.path.join(root, dr)):
                                for f in sorted(files2):
                                    # append data for hours 1-24
                                    if f.endswith('.nc') and 0 < int(f.split('.nc')[0][-3:]) < 25:
                                    # if f.endswith('.nc') and 0 < int(f.split('.nc')[0][-3:]) < 2:  ##### for testing
                                        append_model_data(os.path.join(root2, f), nyserda_buoys, data, height)

                # plot data for each NYSERDA buoy location for each year/division
                if divs == 'monthly':
                    date_div = pd.to_datetime('{}-{:02d}'.format(yr, d)).strftime('%B %Y')
                    sdiv = pd.to_datetime('{}-{:02d}'.format(yr, d)).strftime('%Y%m')

                for loc, da in data.items():
                    if 'north' in loc:
                        buoy = 'NYSERDA North'
                        buoy_code = 'NYNE05'
                    elif 'south' in loc:
                        buoy = 'NYSERDA South'
                        buoy_code = 'NYSE06'
                    n = np.count_nonzero(~np.isnan(da['ws']))
                    ttl = 'RU-WRF 4.1: {}m Wind Rose at {}\n{}, n = {}'.format(str(height), buoy, date_div, n)

                    # plot wind rose
                    sf = 'WRF_windrose_{}_{}.png'.format(buoy_code, sdiv)
                    sfpath = os.path.join(save_dir, sf)
                    plot_windrose(da, ttl, sfpath)


if __name__ == '__main__':
    wrf_dir = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km'  # on server
    # wrf_dir = '/Volumes/boardwalk/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km'
    # wrf_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/website_plots_redo/processed/3km'
    sDir = '/home/lgarzio/rucool/bpu/wrf/windrose'  # on server
    # sDir = '/Users/lgarzio/Documents/rucool/bpu/wrf/windrose'
    years = ['2019', '2020']
    division = 'monthly'
    wsheights = [160]
    main(wrf_dir, sDir, years, division, wsheights)
