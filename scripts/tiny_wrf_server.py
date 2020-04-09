#!/usr/bin/env python
# Tiny WRF Loader! for the creation of smaller WRF files in netCDF Format
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def make_wrf_file(dtime, fo=0, pd=0):
    t2 = dtime.replace()  # Copy variable to mess with
    if pd:
        t2 = t2-timedelta(pd)  # Previous Day
    if t2.hour < fo:
        t2 = t2-timedelta(1)  # Previous model run
        hour = t2.hour + 24
    else:
        hour = t2.hour
    if pd:
        hour = hour+24*pd  # Later in model run
    datestr = '%d%02d%02d' % (t2.year, t2.month, t2.day)
    return '%s/wrfproc_3km_%s_00Z_H%03d.nc' % (datestr, datestr, hour)


def load_wrf(start_date, end_date, forecast_offset, version_num, point_location):
    # directory = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/'  # Server
    if version_num == 'v4.1':
        directory = '/home/coolgroup/ru-wrf/real-time/v4.1_parallel/processed/3km/'  # Server
    elif version_num == 'v3.9':
        directory = '/home/coolgroup/ru-wrf/real-time/processed/3km/'  # Server
    end_date = end_date - timedelta(hours=1)
    times = pd.date_range(start_date, end_date, freq="H")
    heights = np.array([10, 100, 120, 140], dtype='int32')
    sites = pd.read_csv(point_location, skipinitialspace=True)
    stations = sites.name.astype('S')
    data = np.empty(shape=(len(times), len(stations), len(heights))) * np.NAN
    uVel = xr.DataArray(data, coords=[times, stations, heights], dims=['time', 'station', 'height'], attrs={
        'units': 'm s-1',
        'standard_name': 'eastward_wind',
        'long_name': 'Wind Speed, Zonal',
        'comment': 'The zonal wind speed (m/s) indicates the u (positive eastward) component'
                   ' of where the wind is going.',
    })

    uVel['time'].attrs['standard_name'] = 'time'
    uVel['time'].attrs['long_name'] = 'Time'

    uVel['station'].attrs['standard_name'] = 'station_id'
    uVel['station'].attrs['long_name'] = 'Station ID'
    uVel['station'].attrs['comment'] = 'A string specifying a unique station ID, created to allow easy referencing of' \
                                       ' the selected grid points extracted from the WRF model files.'

    uVel['height'].attrs['units'] = 'm'
    uVel['height'].attrs['standard_name'] = 'height'
    uVel['height'].attrs['long_name'] = 'Height'

    vVel = uVel.copy()
    vVel.attrs['standard_name'] = 'northward_wind'
    vVel.attrs['long_name'] = 'Wind Speed, Meridional'
    vVel.attrs['comment'] = 'The meridional wind speed (m/s) indicates the v (positive northward) component of ' \
                            'where the wind is going.'

    latitude = xr.DataArray(sites['latitude'], coords=[stations], dims=['station'], attrs={
        'units': 'degrees_north',
        'comment': 'The latitude of the station.',
        'long_name': 'Latitude',
        'standard_name': 'latitude'
    })
    longitude = xr.DataArray(sites['longitude'], coords=[stations], dims=['station'], attrs={
        'units': 'degrees_east',
        'comment': 'The longitude of the station.',
        'long_name': 'Longitude',
        'standard_name': 'longitude'
    })
    for t in times:
        try:
            wrf_file = make_wrf_file(t, forecast_offset)
            ncdata = xr.open_dataset(directory + wrf_file)
            print('Processing: ' + str(t) + ' File: ' + wrf_file)
            lats = ncdata.XLAT.squeeze()
            lons = ncdata.XLONG.squeeze()

            for index, site in sites.iterrows():
                # Step 4 - Find the closest model point
                a = abs(lats - site.latitude) + abs(lons - site.longitude)
                i, j = np.unravel_index(a.argmin(), a.shape)
                levels = [100, 120, 140]
                hindex = [7, 9, 11]
                uVel.loc[{'time': t, 'station': stations[index], 'height': 10}] = ncdata.U10[0][i][j].item()
                vVel.loc[{'time': t, 'station': stations[index], 'height': 10}] = ncdata.V10[0][i][j].item()
                for ii, jj in zip(levels, hindex):
                    uVel.loc[{'time': t, 'station': stations[index], 'height': ii}] = ncdata.U[0][jj][i][j].item()
                    vVel.loc[{'time': t, 'station': stations[index], 'height': ii}] = ncdata.V[0][jj][i][j].item()
            ncdata.close()

        except:
            print('Could not process file: ' + wrf_file)

    # Wind Speed
    wind_speed = np.sqrt(uVel ** 2 + vVel ** 2)
    wind_speed.attrs['units'] = 'm s-1'
    wind_speed.attrs['comment'] = 'Wind Speed is calculated from the Zonal and Meridional wind speeds.'
    wind_speed.attrs['long_name'] = 'Wind Speed'
    wind_speed.attrs['standard_name'] = 'wind_speed'

    # Wind Direction
    wind_dir = 270 - xr.ufuncs.arctan2(vVel, uVel) * 180 / np.pi
    wind_dir = wind_dir % 360  # Use modulo to keep degrees between 0-360
    wind_dir.attrs['units'] = 'degree'
    wind_dir.attrs['comment'] = 'The direction from which winds are coming from, in degrees clockwise from true N.'
    wind_dir.attrs['long_name'] = 'Wind Direction'
    wind_dir.attrs['standard_name'] = 'wind_from_direction'

    final_dataset = xr.Dataset({
        'u_velocity': uVel, 'v_velocity': vVel,
        'wind_speed': wind_speed, 'wind_dir': wind_dir,
        'latitude': latitude, 'longitude': longitude
    })

    encoding = {}
    encoding['time'] = dict(units='days since 2010-01-01 00:00:00', calendar='gregorian', dtype=np.double)

    return final_dataset


def tiny_wrf2nc(start_date, end_date, buoy, point_location):
    encoding = {}
    encoding['time'] = dict(units='days since 2010-01-01 00:00:00', calendar='gregorian', dtype=np.double)

    wrf_v41_ds = load_wrf(start_date, end_date, 1, 'v4.1', point_location)
    outputfile_41 = buoy[0] + '_' + start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d") + '_41ds.nc'
    wrf_v41_ds.to_netcdf(outputfile_41, encoding=encoding)

    del wrf_v41_ds

    wrf_v39_ds = load_wrf(start_date, end_date, 1, 'v3.9', point_location)
    outputfile_39 = buoy[0] + '_' + start_date.strftime("%Y%m%d") + '_' + end_date.strftime("%Y%m%d") + '_39ds.nc'
    wrf_v39_ds.to_netcdf(outputfile_39, encoding=encoding)


# N'0812 - S'0904
start_date = datetime(2019, 8, 12)
# end_date = datetime(2019, 8, 14)
end_date = datetime(2020, 3, 30)
buoy = 'NYNE05', b'NYNE05'
point_location = 'wrf_validation_lite_points.csv'
tiny_wrf2nc(start_date, end_date, buoy, point_location)

start_date = datetime(2019, 9, 4)
# end_date = datetime(2019, 9, 6)
end_date = datetime(2020, 3, 30)
buoy = 'NYSE06', b'NYSE06'
point_location = 'wrf_validation_lite_points.csv'
tiny_wrf2nc(start_date, end_date, buoy, point_location)

