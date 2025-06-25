#!/usr/bin/env python

from pathlib import Path
import pytest

import numpy as np
import xarray as xr

import clawpack.geoclaw.surge.storm
import clawpack.geoclaw.util as util

# :TODO: Put the storm file creation into a test fixture
# :TODO: Possibly move the mapping test into a test_util file
def test_netcdf_storm_io(tmp_path):
    
    # Create test NetCDF storm file
    storm_data_file = Path(tmp_path) / "storm.nc"
    size = (3, 5, 3)
    coords = [('longitude', np.linspace(-1, 1, size[0])), 
              ('latitude', np.linspace(-2, 2, size[1])), 
              ('valid_time', np.linspace(0, 2, size[2]))]
    wind_x = xr.DataArray(np.random.rand(*size),  coords=coords)
    wind_y = xr.DataArray(np.random.rand(*size), coords=coords)
    P = xr.DataArray(np.random.rand(*size), coords=coords)
    ds = xr.Dataset({'u': wind_x, 'v': wind_y, 'pressure': P,})
    ds.to_netcdf(storm_data_file)

    # Test name finding
    _dim_mapping = util.get_netcdf_names(storm_data_file, 
                                         lookup_type='dim',
                                         verbose=True,
                                         user_mapping={'t': 'valid_time'})
    _var_mapping = util.get_netcdf_names(storm_data_file, 
                                         lookup_type='var',
                                         verbose=True)
    assert _dim_mapping == {'x': 'longitude', 
                            'y': 'latitude', 
                            't': 'valid_time'}
    assert _var_mapping == {'wind_x': 'u', 
                            'wind_y': 'v', 
                            'pressure': 'pressure'}

    # Test storm IO
    storm = clawpack.geoclaw.surge.storm.Storm()
    storm.time_offset = np.datetime64("2012-12-01")
    storm.data_file_format = "netcdf"
    storm.file_paths = [storm_data_file]
    path = tmp_path / Path("test.storm")
    storm.write(path, file_format='netcdf', dim_mapping={'t': 'valid_time'}, 
                      verbose=True)
    read_storm = clawpack.geoclaw.surge.storm.Storm(path, file_format="netcdf")
    assert (storm.time_offset == read_storm.time_offset)
    assert (storm.file_paths == read_storm.file_paths)