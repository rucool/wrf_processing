#!/usr/bin/env python

"""
Author: Lori Garzio on 6/30/2020
Last modified: 6/30/2020
Creates box plots of wind speeds at each hour of the day at user-defined heights from hourly-averaged NYSERDA LiDAR buoy
data. The box limits extend from the lower to upper quartiles, with a line at the median and a diamond symbol at the
mean. Whiskers extend from the box to show the range of the data. Circles indicate outliers.
Data from https://oswbuoysny.resourcepanorama.dnvgl.com/download/f67d14ad-07ab-4652-16d2-08d71f257da1
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console
plt.rcParams.update({'font.size': 14})


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


def main(nys_dir, save_dir, divs, heights):
    # locations and file names of NYSERDA LiDAR buoys
    nyserda_buoys = dict(NYNE05=dict(lon=-72.7173, lat=39.9686, name='NYSERDA North',
                                     fname='E05_Hudson_North_10_min_avg.csv'),
                         NYSE06=dict(lon=-73.4295, lat=39.5465, name='NYSERDA South',
                                     fname='E06_Hudson_South_10_min_avg.csv'))

    hours = np.arange(1, 25, 1)

    for key, item in nyserda_buoys.items():
        nys_ds = pd.read_csv(os.path.join(nys_dir, item['fname']), error_bad_lines=False,
                             delimiter=', ', engine='python')
        nys_ds['timestamp'] = pd.to_datetime(nys_ds['timestamp'])
        for height in heights:
            ws_colname = 'lidar_lidar{}m_Z10_HorizWS'.format(str(height))
            wd_colname = 'lidar_lidar{}m_WD_alg_03'.format(str(height))
            ds = nys_ds[['timestamp', ws_colname, wd_colname]]
            for div in divs:
                if div == 'quarter':
                    start_dates = ['06-01-2019', '09-01-2019', '12-01-2019', '03-01-2020']
                    end_dates = ['08-31-2019', '11-30-2019', '02-29-2020', '05-31-2020']
                elif div == 'year':
                    start_dates = ['06-01-2019']
                    end_dates = ['05-31-2020']
                    # start_dates = ['09-01-2019']  ### for testing
                    # end_dates = ['09-05-2019']  ### for testing

                for t0, t1 in zip(start_dates, end_dates):
                    # initialize empty data dictionary for each hour of day
                    data = dict()
                    for hr in hours:
                        data[hr] = dict(t=np.array([], dtype='datetime64[ns]'), ws=np.array([]), wd=np.array([]))

                    t0_dt = pd.to_datetime(t0)
                    t1_dt = pd.to_datetime(t1)

                    if t1_dt > t0_dt:
                        # check if there are any data for the month before moving to hourly averages
                        ds_check = ds[(t0_dt <= ds['timestamp']) & (ds['timestamp'] <= t1_dt + dt.timedelta(days=1))]
                        if len(ds_check) > 0:
                            # calculate hourly averages for minutes 10-60 and append to data dictionary
                            time_range = pd.date_range(t0_dt + dt.timedelta(hours=1),
                                                       t1_dt + dt.timedelta(days=1), freq='H')
                            for tr in time_range:
                                ds_tr = ds[(tr - dt.timedelta(minutes=50) <= ds['timestamp']) &
                                           (ds['timestamp'] <= tr)]
                                if len(ds_tr) > 0:
                                    meanws = np.nanmean(pd.to_numeric(ds_tr[ws_colname], errors='coerce'))
                                    meanwd = np.nanmean(pd.to_numeric(ds_tr[wd_colname], errors='coerce'))

                                    # if both values aren't NaN, append to dictionary
                                    if ~np.isnan(meanws) and ~np.isnan(meanwd):
                                        hour_key = tr.hour
                                        if hour_key == 0:
                                            hour_key = 24
                                        data[hour_key]['t'] = np.append(data[hour_key]['t'], tr)
                                        data[hour_key]['ws'] = np.append(data[hour_key]['ws'], meanws)
                                        data[hour_key]['wd'] = np.append(data[hour_key]['wd'], meanwd)

                            t0_data = pd.to_datetime(np.nanmin(ds_check['timestamp']))
                            t1_data = pd.to_datetime(np.nanmax(ds_check['timestamp']))
                            if t1_data.hour == 0:
                                t1_data = t1_data - dt.timedelta(hours=1)

                            tm_range_str = '{} to {}'.format(pd.to_datetime(t0_data).strftime('%b %d %Y'),
                                                             pd.to_datetime(t1_data).strftime('%b %d %Y'))
                            sdiv = pd.to_datetime(t0_data).strftime('%Y%m')

                            ttl1 = '{}: {}m'.format(item['name'], str(height))
                            ttl2 = '\n{}'.format(tm_range_str)

                            print('Plotting {}'.format(sdiv))
                            sf = '{}_boxplot_{}m_{}_hrlyavg_{}'.format(key, str(height), sdiv, div)
                            sfpath = os.path.join(save_dir, sf)
                            plot_boxplot(data, ttl1, ttl2, sfpath)


if __name__ == '__main__':
    nyserda_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/data'
    sDir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/boxplot'
    division = ['year']  # ['year', 'quarter']
    wsheights = [158]
    main(nyserda_dir, sDir, division, wsheights)
