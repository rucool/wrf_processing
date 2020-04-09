#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import os
from wrf import getvar, interplevel, default_fill
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def make_broken_wrf_file(dtime, forecast_offset):
    t2 = dtime.replace()
    if t2.hour < forecast_offset:
        day = t2.day - 1
        if day == 0:
            day = day + 1
        # hour = t2.hour + 24
    else:
        day = t2.day
        # hour = t2.hour
    datestr = '%d%02d%02d' % (t2.year, t2.month, day)
    datestr_f = '%d-%02d-%02d_%02d:00:00' % (t2.year, t2.month, t2.day, t2.hour)
    return '%s/wrfout_d02_%s' % (datestr, datestr_f)


def plot_wrf_prof_single(start_date, end_date, prof_h):
    times = pd.date_range(start_date, end_date + timedelta(hours=23), freq="H")
    # fname = 'data/nc_raw/wrfout_d01_2019-12-15_000000.nc'
    # fname = 'temp_data/WVZ2AK~O'

    for t in times:
        # get file name
        fname = make_broken_wrf_file(t)
        path = '/home/coolgroup/ru-wrf/real-time/wrfoutv4.1/'
        # Open using netCDF toolbox
        ncfile_t = xr.open_dataset(path + fname)
        print('processing: ' + fname)
        original_global_attributes = ncfile_t.attrs
        ncfile = ncfile_t._file_obj.ds

        wrf_ws_wd = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")
        wrf_ws = wrf_ws_wd[0, :]
        # wrf_wd = wrf_ws_wd[1, :]

        wrf_loc_ws = wrf_ws[:, 0, 0]
        # wrf_loc_wd = wrf_wd[:, 0, 0]

        # Subtract terrain height from height above sea level
        wrf_z = getvar(ncfile, 'z', units='m') - getvar(ncfile, 'ter', units='m')

        wrf_loc_z = wrf_z[:, 0, 0]

        wrf_loc_z[wrf_loc_z > prof_h] = np.nan

        plt.plot(wrf_loc_ws, wrf_loc_z)
        plt.xlabel('wind speed (m/s)')
        plt.ylabel('height (m)')
        plt.title('WRF 4.1 Wind Profile')
        # plt.show()
        plt.savefig('wind_profile_' + t.strftime() + '_' + str(prof_h) + '.png')
        plt.close()


def plot_wrf_prof_multi(start_time, prof_h, point_location, buoy):
    if buoy[0] == 'NYNE05':
        bind = 2
        buoy_name = 'NYSERDA North'
    elif buoy[0] == 'NYSE06':
        bind = 3
        buoy_name = 'NYSERDA South'
    else:
        print('incorrect buoy name')

    times = pd.date_range(start_time, start_time + timedelta(hours=23), freq="H")
    sites = pd.read_csv(point_location, skipinitialspace=True)

    # NYSERDA LOAD
    nyserda_ds = load_nyserda_rework_fullprofile(buoy)
    nyserda_ds.reset_index(inplace=True)
    # nyserda_ds[nyserda_ds['index'].isin(times)] # full matrix

    # MODEL LOAD
    nam_ds_fullprof = load_nam_fullprof(start_time, end_date, buoy, point_location)
    gfs_ds_fullprof = load_gfs_fullprof(start_time, end_date, buoy, point_location)
    hrrr_ds_fullprof = load_hrrr_fullprof(start_time, end_date, buoy, point_location)

    for ind, t in enumerate(times):
        my_dpi = 96
        plt.figure(figsize=(600 / my_dpi, 500 / my_dpi), dpi=my_dpi)
        # get file name
        # fname = 'temp_data/WVZ2AK~O'
        fname = make_broken_wrf_file(t, forecast_offset=1)
        path = '/home/coolgroup/ru-wrf/real-time/wrfoutv4.1/'
        save_path = '/home/jad438/wrf_wp/' + t.strftime('%Y_%m') + '_big4/'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # NYSERDA LINE
        nyserda_ws = nyserda_ds[nyserda_ds['index'] == times[ind]].values[:, 1:]
        nyserda_hs = np.arange(18, 218, 20)
        line1 = plt.plot(nyserda_ws[0][:], nyserda_hs, '.-', color='black', label='NYSERDA', linewidth=3)

        # NAM LINE
        nam_ws = nam_ds_fullprof[nam_ds_fullprof['timestamp'] == times[ind] - timedelta(hours=1)].values[:, 1:]
        nam_hs = [10, 80]
        line2 = plt.plot(nam_ws[0][:], nam_hs, '.-', label='NAM')
        # print(times[ind] - timedelta(hours=1))

        # GFS LINE
        gfs_ws = gfs_ds_fullprof[gfs_ds_fullprof['timestamp'] == times[ind] - timedelta(hours=1)].values[:, 1:]
        gfs_hs = [10, 30, 40, 50, 80, 100]
        line3 = plt.plot(gfs_ws[0][:], gfs_hs, '.-', label='GFS')

        # HRRR LINE
        hrrr_ws = hrrr_ds_fullprof[hrrr_ds_fullprof['timestamp'] == times[ind] - timedelta(hours=1)].values[:, 1:]
        hrrr_hs = [10, 80]
        line4 = plt.plot(hrrr_ws[0][:], hrrr_hs, '.-', label='HRRR')

        # Open using netCDF toolbox
        ncfile_t = xr.open_dataset(path + fname)
        print('processing: ' + fname)
        original_global_attributes = ncfile_t.attrs
        ncfile = ncfile_t._file_obj.ds

        # Get variables and subtract terrain height from height above sea level
        wrf_z = getvar(ncfile, 'z', units='m') - getvar(ncfile, 'ter', units='m')
        wrf_XLAT = getvar(ncfile, 'XLAT').squeeze()
        wrf_XLON = getvar(ncfile, 'XLONG').squeeze()

        wrf_ws_wd = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")
        wrf_ws = wrf_ws_wd[0, :]
        wrf_wd = wrf_ws_wd[1, :]

        # Find the closest model point
        a = abs(wrf_XLAT - sites['latitude'][bind]) + abs(wrf_XLON - sites['longitude'][bind])
        i, j = np.unravel_index(a.argmin(), a.shape)
        wrf_loc_ws = wrf_ws[:, i, j]
        wrf_loc_wd = wrf_wd[:, i, j]
        wrf_loc_z = wrf_z[:, i, j]

        # plotting
        wrf_loc_z[wrf_loc_z > prof_h] = np.nan
        line5 = plt.plot(wrf_loc_ws, wrf_loc_z, '.-', linewidth=3, label='WRF 4.1')

        plt.axhline(y=40, color='gray', linestyle='--')
        plt.axhline(y=240, color='gray', linestyle='--')
        plt.legend(loc='best')
        plt.xlabel('wind speed (m/s)')
        plt.ylabel('height (m)')
        plt.title('Model Wind Profiles ' + buoy_name + ' ' + t.strftime('%Y/%m/%d %HH'))
        # plt.show()
        plt.savefig(
            '/home/jad438/wrf_wp/' + t.strftime('%Y_%m') + '_big4/' + 'wind_profiles_' + buoy[0] + '_' +
            str(prof_h) + 'm_' + t.strftime('%Y%m%d_%H') + 'H_big4.png')
        print('figure complete')
        plt.close()


def plot_wrf_prof_subplot(start_date, end_date, prof_h, point_location):
    # fname = 'data/nc_raw/wrfout_d01_2019-12-15_000000.nc'
    fname = 'temp_data/WVZ2AK~O'
    sites = pd.read_csv(point_location, skipinitialspace=True)
    # stations = sites.name.astype('S')

    # my_dpi = 96
    # plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    # get file name
    # Open using netCDF toolbox
    ncfile_t = xr.open_dataset(fname)
    print('processing: ' + fname)
    original_global_attributes = ncfile_t.attrs
    ncfile = ncfile_t._file_obj.ds

    wrf_XLAT = getvar(ncfile, 'XLAT').squeeze()
    wrf_XLON = getvar(ncfile, 'XLONG').squeeze()
    # Subtract terrain height from height above sea level
    wrf_z = getvar(ncfile, 'z', units='m') - getvar(ncfile, 'ter', units='m')

    wrf_ws_wd = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")
    wrf_ws = wrf_ws_wd[0, :]
    wrf_wd = wrf_ws_wd[1, :]

    # Find the closest model point
    a = abs(wrf_XLAT - sites['latitude'][2]) + abs(wrf_XLON - sites['longitude'][2])
    i, j = np.unravel_index(a.argmin(), a.shape)
    wrf_loc_ws_NYN = wrf_ws[:, i, j]
    wrf_loc_wd_NYN = wrf_wd[:, i, j]
    wrf_loc_z_NYN = wrf_z[:, i, j]

    a = abs(wrf_XLAT - sites['latitude'][3]) + abs(wrf_XLON - sites['longitude'][3])
    i, j = np.unravel_index(a.argmin(), a.shape)
    wrf_loc_ws_NYS = wrf_ws[:, i, j]
    wrf_loc_wd_NYS = wrf_wd[:, i, j]
    wrf_loc_z_NYS = wrf_z[:, i, j]

    # wrf_loc_ws = wrf_ws[:, 0, 0]
    # wrf_loc_wd = wrf_wd[:, 0, 0]
    # wrf_loc_z = wrf_z[:, 0, 0]

    wrf_loc_z_NYN[wrf_loc_z_NYN > prof_h] = np.nan
    wrf_loc_z_NYS[wrf_loc_z_NYS > prof_h] = np.nan
    n = 24
    # color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
    # for i in range(n):
    #     c = next(color)
    #     plt.plot(x, y, c=c)
    # fig, ax = plt.subplot()

    colors = plt.cm.rainbow(np.linspace(0, 1, n))

    my_dpi = 96
    fig, axs = plt.subplots(2, 1, figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)

    axs[0].plot(wrf_loc_ws_NYN, wrf_loc_z_NYN, color=colors[0])
    axs[1].plot(wrf_loc_ws_NYS, wrf_loc_z_NYS, color=colors[0])

    axs[0].set_ylim(0, 260)
    axs[1].set_ylim(0, 260)

    axs[0].set_xlabel('wind speed (m/s)')
    axs[1].set_xlabel('wind speed (m/s)')
    axs[0].set_ylabel('height (m)')
    axs[1].set_ylabel('height (m)')

    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].title.set_text('WRF 4.1 Wind Profile at NYSERDA North')
    axs[1].title.set_text('WRF 4.1 Wind Profile at NYSERDA South')

    fig = plt.gcf()
    ax = plt.gca()

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'),
                                            ax=ax, orientation='horizontal', fraction=0.09, pad=0.25)
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.ax.set_xticklabels(['0H', '6H', '12H', '18PM', '24H'])

    fig.tight_layout()
    plt.show()
    # plt.savefig('wind_profile_test_' + str(prof_h) + '.png')
    plt.close()


def plot_wrf_prof(start_date, end_date, prof_h, point_location, nyserda_buoy):
    if nyserda_buoy == 'North_E05':
        bind = 2
        buoy_name = 'NYSERDA North'
    elif nyserda_buoy == 'South_E06':
        bind = 3
        buoy_name = 'NYSERDA South'
    else:
        print('incorrect buoy name')

    # fname = 'data/nc_raw/wrfout_d01_2019-12-15_000000.nc'
    fname = 'temp_data/WVZ2AK~O'
    sites = pd.read_csv(point_location, skipinitialspace=True)
    # stations = sites.name.astype('S')
    # my_dpi = 96
    # plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    # get file name
    # Open using netCDF toolbox
    ncfile_t = xr.open_dataset(fname)
    print('processing: ' + fname)
    original_global_attributes = ncfile_t.attrs
    ncfile = ncfile_t._file_obj.ds

    wrf_XLAT = getvar(ncfile, 'XLAT').squeeze()
    wrf_XLON = getvar(ncfile, 'XLONG').squeeze()
    # Subtract terrain height from height above sea level
    wrf_z = getvar(ncfile, 'z', units='m') - getvar(ncfile, 'ter', units='m')

    wrf_ws_wd = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")
    wrf_ws = wrf_ws_wd[0, :]
    wrf_wd = wrf_ws_wd[1, :]

    # Find the closest model point
    a = abs(wrf_XLAT - sites['latitude'][bind]) + abs(wrf_XLON - sites['longitude'][bind])
    i, j = np.unravel_index(a.argmin(), a.shape)
    wrf_loc_ws = wrf_ws[:, i, j]
    wrf_loc_wd = wrf_wd[:, i, j]
    wrf_loc_z = wrf_z[:, i, j]

    wrf_loc_z[wrf_loc_z > prof_h] = np.nan

    n = 24
    colors = plt.cm.rainbow(np.linspace(0, 1, n))

    my_dpi = 96
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    fig = plt.gcf()
    axs = plt.gca()

    axs.plot(wrf_loc_ws, wrf_loc_z, '.-', color=colors[0])

    axs.set_ylim(0, 260)

    axs.set_xlabel('wind speed (m/s)')
    axs.set_ylabel('height (m)')

    axs.grid(True)

    axs.title.set_text('WRF 4.1 Wind Profile at ' + buoy_name)

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'),
                        ax=axs, orientation='horizontal', fraction=0.09, pad=0.25)
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.ax.set_xticklabels(['0H', '6H', '12H', '18PM', '24H'])

    fig.tight_layout()
    plt.show()
    # plt.savefig('wind_profile_test_' + str(prof_h) + '.png')
    plt.close()


def plot_wrf_prof_daily(start_date, prof_h, point_location, nyserda_buoy):
    if nyserda_buoy == 'North_E05':
        bind = 2
        buoy_name = 'NYSERDA North'
    elif nyserda_buoy == 'South_E06':
        bind = 3
        buoy_name = 'NYSERDA South'
    else:
        print('incorrect buoy name')

    times = pd.date_range(start_date, start_date + timedelta(hours=23), freq="H")
    sites = pd.read_csv(point_location, skipinitialspace=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, 24))
    my_dpi = 96
    plt.figure(figsize=(800 / my_dpi, 600 / my_dpi), dpi=my_dpi)
    max_ws = []
    min_ws = []

    for ind, t in enumerate(times):
        # get file name
        fname = make_broken_wrf_file(t, forecast_offset=1)
        path = '/home/coolgroup/ru-wrf/real-time/wrfoutv4.1/'
        save_path = '/home/jad438/wrf_wp/' + t.strftime('%Y_%m') + '_daily/'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Open using netCDF toolbox
        ncfile_t = xr.open_dataset(path + fname)
        print('processing: ' + fname)
        original_global_attributes = ncfile_t.attrs
        ncfile = ncfile_t._file_obj.ds

        wrf_XLAT = getvar(ncfile, 'XLAT').squeeze()
        wrf_XLON = getvar(ncfile, 'XLONG').squeeze()
        # Subtract terrain height from height above sea level
        wrf_z = getvar(ncfile, 'z', units='m') - getvar(ncfile, 'ter', units='m')

        wrf_ws_wd = getvar(ncfile, "uvmet_wspd_wdir", units="m s-1")
        wrf_ws = wrf_ws_wd[0, :]
        wrf_wd = wrf_ws_wd[1, :]

        # Find the closest model point
        a = abs(wrf_XLAT - sites['latitude'][bind]) + abs(wrf_XLON - sites['longitude'][bind])
        i, j = np.unravel_index(a.argmin(), a.shape)
        wrf_loc_ws = wrf_ws[:, i, j]
        wrf_loc_wd = wrf_wd[:, i, j]
        wrf_loc_z = wrf_z[:, i, j]

        wrf_loc_z[wrf_loc_z > prof_h] = np.nan
        # print(max(wrf_loc_ws.data))
        max_ws.append((ind, max(wrf_loc_ws.data)))
        # print(min(wrf_loc_ws.data))
        min_ws.append((ind, min(wrf_loc_ws.data)))
        plt.plot(wrf_loc_ws, wrf_loc_z, '.-', color=colors[ind])

    plt.axhline(y=40, color='gray', linestyle='--')
    plt.axhline(y=240, color='gray', linestyle='--')
    plt.xlabel('wind speed (m/s)')
    plt.ylabel('height (m)')
    plt.title('WRF 4.1 Wind Profiles ' + t.strftime('%Y/%m/%d'))
    fig = plt.gcf()
    ax = plt.gca()
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='rainbow'), ax=ax, orientation='horizontal',
                        fraction=0.05, pad=0.1)
    # cbar.ax.set_yticklabels(pd.to_datetime(cbar.ax.get_yticks()).strftime(date_format='%Y-%m-%d'))
    # cbar.ax.set_xticklabels(times.strftime(date_format='%I%p'))
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.ax.set_xticklabels(['0H', '6H', '12H', '18PM', '24H'])
    # plt.show()
    plt.savefig(
        '/home/jad438/wrf_wp/' + t.strftime('%Y_%m') + '_daily/' + 'wind_profiles_' + nyserda_buoy + '_' +
        str(prof_h) + 'm_' + t.strftime('%Y%m%d') + '.png')
    print('figure complete')
    plt.close()


def load_nyserda_rework_fullprofile(buoy):
    if buoy[0] == 'NYNE05':
        # filename = 'E05_Hudson_North_10_min_avg_20190812_20191006.dat'
        filename = 'E05_Hudson_North_10_min_avg_20190812_20200112.csv'
        start_date = datetime(2019, 8, 12)
        end_date = datetime(2020, 1, 13)
    elif buoy[0] == 'NYSE06':
        # filename = 'E06_Hudson_South_10_min_avg_20190904_20191006.dat'
        filename = 'E06_Hudson_South_10_min_avg_20190904_20200112.csv'
        start_date = datetime(2019, 9, 4)
        end_date = datetime(2020, 1, 13)
    else:
        print('Not a correct buoy')

    # dir = '/Users/jadendicopoulos/Downloads/NYSERDA Floating LiDAR Buoy Data/'
    dir = '/home/jad438/data/NYSERDA Floating LiDAR Buoy Data/'
    nys_ds = pd.read_csv(dir + filename, error_bad_lines=False, delimiter=', ')  # , delim_whitespace=True)

    nys_ws_1hr_nonav = nys_ds  # data every 6 steps (1 hour)
    nys_ws_1hr_nonav = nys_ws_1hr_nonav.reset_index()

    time = pd.date_range(start_date, end_date-timedelta(hours=1), freq='H')

    nys_ws_time = pd.to_datetime(nys_ws_1hr_nonav['timestamp'], format='%m-%d-%Y %H:%M')

    nys_ws_1hr_nonav['lidar_lidar18m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar18m_Z10_HorizWS']
                                                                   , errors='coerce')
    nys_ws_18m = pd.Series(nys_ws_1hr_nonav['lidar_lidar18m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar38m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar38m_Z10_HorizWS']
                                                                   , errors='coerce')
    nys_ws_38m = pd.Series(nys_ws_1hr_nonav['lidar_lidar38m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar58m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar58m_Z10_HorizWS']
                                                                   , errors='coerce')
    nys_ws_58m = pd.Series(nys_ws_1hr_nonav['lidar_lidar58m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar78m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar78m_Z10_HorizWS']
                                                                   , errors='coerce')
    nys_ws_78m = pd.Series(nys_ws_1hr_nonav['lidar_lidar78m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar98m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar98m_Z10_HorizWS']
                                                                   , errors='coerce')
    nys_ws_98m = pd.Series(nys_ws_1hr_nonav['lidar_lidar98m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar118m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar118m_Z10_HorizWS']
                                                                    , errors='coerce')
    nys_ws_118m = pd.Series(nys_ws_1hr_nonav['lidar_lidar118m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar138m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar138m_Z10_HorizWS']
                                                                    , errors='coerce')
    nys_ws_138m = pd.Series(nys_ws_1hr_nonav['lidar_lidar138m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS']
                                                                    , errors='coerce')
    nys_ws_158m = pd.Series(nys_ws_1hr_nonav['lidar_lidar158m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar178m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar178m_Z10_HorizWS']
                                                                    , errors='coerce')
    nys_ws_178m = pd.Series(nys_ws_1hr_nonav['lidar_lidar178m_Z10_HorizWS'].values)
    nys_ws_1hr_nonav['lidar_lidar198m_Z10_HorizWS'] = pd.to_numeric(nys_ws_1hr_nonav['lidar_lidar198m_Z10_HorizWS']
                                                                    , errors='coerce')
    nys_ws_198m = pd.Series(nys_ws_1hr_nonav['lidar_lidar198m_Z10_HorizWS'].values)

    frame = {'timestamp': nys_ws_time, '18m': nys_ws_18m, '38m': nys_ws_38m, '58m': nys_ws_58m, '78m': nys_ws_78m
             , '98m': nys_ws_98m, '118m': nys_ws_118m, '138m': nys_ws_138m, '158m': nys_ws_158m, '178m': nys_ws_178m
             , '198m': nys_ws_198m}
    wrf_fullprof = pd.DataFrame(frame)

    nys_ws_1hr_nonav = wrf_fullprof.set_index('timestamp').reindex(time)

    return nys_ws_1hr_nonav


def load_nam_fullprof(start_date, end_date, buoy, point_location):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=23), freq='D')  # -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/namdata/'
    nam_ws_10 = []
    nam_ws_80 = []
    nam_dt = np.empty((0,), dtype='datetime64[m]')

    for ind, date in enumerate(time_span_D):
        file = 'nams_data_' + date.strftime("%Y%m%d") + '.nc'
        nam_ds = xr.open_dataset(directory + file)
        lats = nam_ds.gridlat_0.squeeze()
        lons = nam_ds.gridlon_0.squeeze()
        site_code = sites[sites['name'] == buoy[0]].index[0]
        a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
        i, j = np.unravel_index(a.argmin(), a.shape)
        nam_ws_10 = np.append(nam_ws_10, nam_ds.wind_speed[:, 0, i, j])
        nam_ws_80 = np.append(nam_ws_80, nam_ds.wind_speed[:, 1, i, j])
        nam_dt = np.append(nam_dt, nam_ds.time)

        frame = {'timestamp': nam_dt, '10m': nam_ws_10, '80m': nam_ws_80}
        nam_ds_fullprof = pd.DataFrame(frame)

        return nam_ds_fullprof


def load_hrrr_fullprof(start_date, end_date, buoy, point_location):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=23), freq='D')  # -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/hrrrdata/'
    hrrr_ws_10 = []
    hrrr_ws_80 = []
    hrrr_dt = np.empty((0,), dtype='datetime64[m]')

    for ind, date in enumerate(time_span_D):
        file = 'hrrr_data_' + date.strftime("%Y%m%d") + '.nc'
        hrrr_ds = xr.open_dataset(directory + file)
        lats = hrrr_ds.gridlat_0.squeeze()
        lons = hrrr_ds.gridlon_0.squeeze()
        site_code = sites[sites['name'] == buoy[0]].index[0]
        a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
        i, j = np.unravel_index(a.argmin(), a.shape)
        hrrr_ws_10 = np.append(hrrr_ws_10, hrrr_ds.wind_speed[:, 0, i, j])
        hrrr_ws_80 = np.append(hrrr_ws_80, hrrr_ds.wind_speed[:, 1, i, j])
        hrrr_dt = np.append(hrrr_dt, hrrr_ds.time)

        frame = {'timestamp': hrrr_dt, '10m': hrrr_ws_10, '80m': hrrr_ws_80}
        hrrr_ds_fullprof = pd.DataFrame(frame)

        return hrrr_ds_fullprof


def load_gfs_fullprof(start_date, end_date, buoy, point_location):
    sites = pd.read_csv(point_location, skipinitialspace=True)
    time_span_D = pd.date_range(start_date, end_date-timedelta(hours=23), freq='D')  # -timedelta(days=1) was removed from here
    directory = '/home/jad438/validation_data/gfsdata/'
    gfs_ws_10 = []
    gfs_ws_30 = []
    gfs_ws_40 = []
    gfs_ws_50 = []
    gfs_ws_80 = []
    gfs_ws_100 = []
    gfs_dt = np.empty((0,), dtype='datetime64[m]')

    for ind, date in enumerate(time_span_D):
        file = 'gfs_data_' + date.strftime("%Y%m%d") + '.nc'
        gfs_ds = xr.open_dataset(directory + file)
        lats = gfs_ds.lat_0.squeeze()
        lons = gfs_ds.lon_0.squeeze()-360
        site_code = sites[sites['name'] == buoy[0]].index[0]
        a = abs(lats - sites.latitude[site_code]) + abs(lons - sites.longitude[site_code])
        i, j = np.unravel_index(a.argmin(), a.shape)
        gfs_ws_10 = np.append(gfs_ws_10, gfs_ds.wind_speed[:, 0, i, j])
        gfs_ws_30 = np.append(gfs_ws_30, gfs_ds.wind_speed[:, 1, i, j])
        gfs_ws_40 = np.append(gfs_ws_40, gfs_ds.wind_speed[:, 2, i, j])
        gfs_ws_50 = np.append(gfs_ws_50, gfs_ds.wind_speed[:, 3, i, j])
        gfs_ws_80 = np.append(gfs_ws_80, gfs_ds.wind_speed[:, 4, i, j])
        gfs_ws_100 = np.append(gfs_ws_100, gfs_ds.wind_speed[:, 5, i, j])
        gfs_dt = np.append(gfs_dt, gfs_ds.time)

        frame = {'timestamp': gfs_dt, '10m': gfs_ws_10, '30m': gfs_ws_30, '40m': gfs_ws_40
                 , '50m': gfs_ws_50, '80m': gfs_ws_80, '100m': gfs_ws_100}
        gfs_ds_fullprof = pd.DataFrame(frame)

        return gfs_ds_fullprof


start_date = datetime(2019, 10, 2)
end_date = datetime(2019, 10, 3)
point_location = 'wrf_validation_lite_points.csv'
# buoy = 'NYNE05', b'NYNE05'
buoy = 'NYSE06', b'NYSE06'
# plot_wrf_prof(start_date, end_date, prof_h=20000)
# plot_wrf_prof_multi(start_date, prof_h=260, point_location=point_location, buoy=buoy)
times = pd.date_range(start_date + timedelta(hours=1), end_date + timedelta(hours=24), freq="D")
for ii in range(0, len(times)):
    plot_wrf_prof_multi(times[ii], prof_h=260, point_location=point_location, buoy=buoy)

# test = load_nyserda_rework_fullprofile(buoy)

# start_date = datetime(2019, 12, 30)
# end_date = datetime(2020, 1, 1)
# # end_date = datetime(2020, 1, 1)
# # plot_wrf_prof_daily(start_date, end_date, prof_h=20000)
# point_location = 'wrf_validation_lite_points.csv'
# nyserda_buoy = 'South_E06'
# times = pd.date_range(start_date + timedelta(hours=1), end_date + timedelta(hours=24), freq="D")
# for ii in range(0, len(times)):
#     plot_wrf_prof_daily(times[ii], prof_h=260, point_location=point_location, nyserda_buoy=nyserda_buoy)

