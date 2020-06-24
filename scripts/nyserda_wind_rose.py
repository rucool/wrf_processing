#!/usr/bin/env python

"""
Author: Lori Garzio on 6/24/2020
Last modified: 6/24/2020
Creates hourly averaged wind rose plots for user-defined time ranges and heights from the NYSERDA LiDAR buoys. Data from
https://oswbuoysny.resourcepanorama.dnvgl.com/download/f67d14ad-07ab-4652-16d2-08d71f257da1
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes
pd.set_option('display.width', 320, "display.max_columns", 10)  # for display in pycharm console


def new_axes():
    """
    Create new wind rose axes
    """
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='w')
    rect = [0.15, 0.15, 0.75, 0.75]
    ax = WindroseAxes(fig, rect, facecolor='w')
    fig.add_axes(ax)
    return ax


def plot_windrose(ws, wd, plt_ttl, save_filepath):
    """
    Wind rose plots of wind speed and direction
    ws: wind speed data
    wd: wind direction data
    plt_ttl: plot title
    save_filepath: full file path to save directory and save filename
    """
    # ccodes = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']  # yellow blue
    # ccodes = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026']  # yellow red
    # ccodes = ['#c7e9b4', '#41b6c4', '#756bb1', '#54278f', '#fc8d59', '#b30000']  # blues purples reds

    ax = new_axes()

    # set the bins for wind speeds
    b = [0, 5, 10, 15, 20, 25, 30]
    # ax.bar(wd, data_dict['ws'], normed=True, bins=b, opening=1, edgecolor='black', colors=ccodes, nsector=36)
    # ax.bar(wd, data_dict['ws'], normed=True, opening=1, edgecolor='black', cmap=cm.viridis, nsector=36)
    ax.bar(wd, ws, normed=True, bins=b, opening=1, edgecolor='black', cmap=cm.jet, nsector=36)

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


def main(nys_dir, yrs, divs, heights):
    # locations and file names of NYSERDA LiDAR buoys
    nyserda_buoys = dict(NYNE05=dict(lon=-72.7173, lat=39.9686, name='NYSERDA North',
                                     fname='E05_Hudson_North_10_min_avg.csv'),
                         NYSE06=dict(lon=-73.4295, lat=39.5465, name='NYSERDA South',
                                     fname='E06_Hudson_South_10_min_avg.csv'))

    for key, item in nyserda_buoys.items():
        nys_ds = pd.read_csv(os.path.join(nys_dir, item['fname']), error_bad_lines=False,
                             delimiter=', ', engine='python')
        nys_ds['timestamp'] = pd.to_datetime(nys_ds['timestamp'])
        for height in heights:
            ws_colname = 'lidar_lidar{}m_Z10_HorizWS'.format(str(height))
            wd_colname = 'lidar_lidar{}m_WD_alg_03'.format(str(height))
            ds = nys_ds[['timestamp', ws_colname, wd_colname]]
            for yr in yrs:
                if divs == 'monthly':
                    dt_start = dt.datetime.strptime('1-1-{}'.format(yr), '%m-%d-%Y')
                    dt_end = dt.datetime.strptime('12-31-{}'.format(yr), '%m-%d-%Y')
                    start_dates = [dt_start.strftime('%m-%d-%Y')]
                    end_dates = []
                    ts1 = dt_start
                    while ts1 <= dt_end:
                        ts2 = ts1 + dt.timedelta(days=1)
                        if ts2.month != ts1.month:
                            start_dates.append(ts2.strftime('%m-%d-%Y'))
                            end_dates.append(ts1.strftime('%m-%d-%Y'))
                        ts1 = ts2

                    end_dates.append(dt_end.strftime('%m-%d-%Y'))

                    for t0, t1 in zip(start_dates, end_dates):
                        # initialize empty data dictionary
                        data = dict(t=np.array([], dtype='datetime64[ns]'), ws=np.array([]), wd=np.array([]))

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
                                            data['t'] = np.append(data['t'], tr)
                                            data['ws'] = np.append(data['ws'], meanws)
                                            data['wd'] = np.append(data['wd'], meanwd)

                                t0_data = pd.to_datetime(np.nanmin(ds_check['timestamp']))
                                t1_data = pd.to_datetime(np.nanmax(ds_check['timestamp']))
                                if t1_data.hour == 0:
                                    t1_data = t1_data - dt.timedelta(hours=1)

                                t0_datastr = t0_data.strftime('%Y-%m-%d')
                                t1_datastr = t1_data.strftime('%Y-%m-%d')
                                ttl = '{} {}m Wind Rose\n{} to {}'.format(item['name'], str(height), t0_datastr, t1_datastr)

                                t0_save_str = t0_data.strftime('%Y%m')
                                print('Plotting {}'.format(t0_save_str))
                                sf = '{}_windrose_{}_{}_hrlyavg.png'.format(key, t0_save_str, divs)
                                sfpath = os.path.join(os.path.dirname(nys_dir), sf)
                                plot_windrose(data['ws'], data['wd'], ttl, sfpath)


if __name__ == '__main__':
    nyserda_dir = '/Users/lgarzio/Documents/rucool/bpu/wrf/nyserda/data'
    years = ['2019', '2020']
    division = 'monthly'
    wsheights = [158]
    main(nyserda_dir, years, division, wsheights)
