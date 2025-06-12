#!/usr/bin/env python

from pathlib import Path
import pytest

import numpy as np

import clawpack.geoclaw.surge.storm

def test_netcdf_storm_io(tmp_path):
    storm = clawpack.geoclaw.surge.storm.Storm()
    storm.time_offset = np.datetime64("2012-12-01")
    storm.data_file_format = "netcdf"
    storm.file_paths = [Path("storm.nc")]
    path = tmp_path / Path("test.storm")
    storm.write(path, file_format='netcdf')
    read_storm = clawpack.geoclaw.surge.storm.Storm(path, 
                                                    file_format="netcdf")
    assert (storm.time_offset == read_storm.time_offset)
    assert (storm.file_paths == read_storm.file_paths)