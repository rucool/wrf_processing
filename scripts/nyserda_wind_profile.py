#!/usr/bin/env python

"""
Author: Lori Garzio on 6/23/2020
Last modified: 6/23/2020
Creates profile plots of hourly averaged wind speeds from the NYSERDA LiDAR buoys. Data from
https://oswbuoysny.resourcepanorama.dnvgl.com/download/f67d14ad-07ab-4652-16d2-08d71f257da1
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
plt.rcParams.update({'font.size': 14})  # set the font size for all plots


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
    cbar_labs = ['01:00', '06:00', '12:00', '18:00', '24:00']

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(right=0.88, left=0.15)
    plt.grid()
    for i in range(n):
        if hmax is not None:
            height_ind = np.where(data_dict['height'] <= hmax)
            ax.plot(data_dict['ws'][i][height_ind], data_dict['height'][height_ind], c=colors[i])
        else:
            ax.plot(data_dict['ws'][i], data_dict['height'], c=colors[i])
        if i == (n - 1):
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'),
                                ax=ax, orientation='vertical', fraction=0.09, pad=0.03, label='Hour (GMT)')
            cbar.set_ticks([0, .25, .5, .75, 1])
            cbar.ax.set_yticklabels(cbar_labs)
            ax.set_xlabel('Hourly Averaged Wind Speed (m/s)')
            ax.set_ylabel('Height (m)')
            ax.set_title(plt_ttl)
            ax.set_xlim(0, 30)

            plt.savefig(save_filepath, dpi=200)
            plt.close()


def main(nys_dir, buoys, sdate, edate):
    # locations and file names of NYSERDA LiDAR buoys
    nyserda_buoys = dict(NYNE05=dict(lon=-72.7173, lat=39.9686, name='NYSERDA North',
                                     fname='E05_Hudson_North_10_min_avg.csv'),
                         NYSE06=dict(lon=-73.4295, lat=39.5465, name='NYSERDA South',
                                     fname='E06_Hudson_South_10_min_avg.csv'))

    dates = pd.date_range(sdate, edate, freq='D')

    for buoy in buoys:
        nys_ds = pd.read_csv(os.path.join(nys_dir, nyserda_buoys[buoy]['fname']), error_bad_lines=False,
                             delimiter=', ', engine='python')
        cols = [x for x in nys_ds.columns if '_HorizWS' in x]
        nys_ds['timestamp'] = pd.to_datetime(nys_ds['timestamp'])
        for date in dates:
            time_range = pd.date_range(date + dt.timedelta(hours=1), date + dt.timedelta(days=1), freq='H')

            # initialize empty data dictionary
            data = dict(t=np.array([], dtype='datetime64[ns]'), height=np.array([]), ws=np.array([]))

            # for each hour in the date, calculate hourly averages for minutes 10-60 and append to data dictionary
            for tr in time_range:
                heights = []
                windspeeds = []
                ds = nys_ds[(tr - dt.timedelta(minutes=50) <= nys_ds['timestamp']) & (nys_ds['timestamp'] <= tr)]
                for col in cols:
                    heights.append(int(col.split('_')[1].split('lidar')[-1].split('m')[0]))
                    meanws = np.nanmean(pd.to_numeric(ds[col], errors='coerce'))
                    windspeeds.append(meanws)

                # append data to dictionary
                data['t'] = np.append(data['t'], tr)
                if len(data['height']) > 0:
                    data['ws'] = np.vstack((data['ws'], windspeeds))
                else:
                    data['height'] = heights
                    data['ws'] = windspeeds

            # plot data
            ttl = '{} Wind Profiles: {}\nhourly averages for minutes 10-60'.format(nyserda_buoys[buoy]['name'],
                                                                                   date.strftime('%Y-%m-%d'))
            sf = 'NYSERDA_{}_wsprofiles_{}.png'.format(buoy, date.strftime('%Y%m%d'))
            sfpath = os.path.join(os.path.dirname(nys_dir), sf)
            plot_wndsp_profile(data, ttl, sfpath)


if __name__ == '__main__':
    nyserda_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/data'
    buoys = ['NYNE05', 'NYSE06']
    start_date = dt.datetime(2020, 5, 28)
    end_date = dt.datetime(2020, 5, 29)
    main(nyserda_dir, buoys, start_date, end_date)
