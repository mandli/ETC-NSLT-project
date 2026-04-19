#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np

import batch.batch
from clawpack.geoclaw.surge.storm import Storm
import clawpack.clawutil as clawutil
import clawpack.geoclaw.util as util

scratch_dir = os.path.join(os.environ["CLAW"], 'geoclaw', 'scratch')

str_val = lambda value: str(int(value * 10)).zfill(2)

class ETCJob(batch.batch.Job):

    def __init__(self, storm_path, article=False, scaling=1.0, 
                       sea_level=0.0, levels=2):

        super(ETCJob, self).__init__()

        self.scaling = scaling

        self.type = "surge"
        self.name = "ETC_storms"
        self.prefix = f"WS{str_val(scaling)}_SL{str_val(sea_level)}_L{levels}"
        self.prefix += f"_{storm_path.stem}"
        self.executable = "xgeoclaw"

        # Create base data object
        import setrun
        self.rundata = setrun.setrun()
        self.rundata.amrdata.amr_levels_max = levels
        self.rundata.geo_data.sea_level = sea_level

        # Point surge.data to the appropriate storm file
        # TODO: Check to if the path to the data already exists
        self.rundata.surge_data.storm_file = (Path(os.environ['DATA_PATH'])
                                / "surge" / "ETC_storms" / f"{self.prefix}_data"
                                / f"{self.prefix}.storm").resolve()
        
        # Path to storm data
        self.storm_path = storm_path

    def __str__(self):
        output = super(ETCJob, self).__str__()
        output += f"  storm path: {self.storm_path}\n"
        output += f"  Scaling: {self.scaling}\n"
        output += f"  Sea-Level: {self.rundata.geo_data.sea_level}\n"
        output += f"  Max Levels: {self.rundata.amrdata.amr_levels_max}\n"
        return output


    def write_data_objects(self):
        r""""""

        # Write out storm
        etc_storm = Storm()
        # Wrap coordinates
        # input_path = (storm_path).resolve()
        # output_path = ().resolve()
        # util.wrap_coords(input_path, output_path=output_path,
                                    # dim_mapping={'t': 'valid_time'})
        etc_storm.file_paths.append(self.storm_path)
        # etc_storm.time_offset = np.datetime64("2012-12-26")
        etc_storm.time_offset = np.datetime64("2018-11-14T08:00:00.00")
        etc_storm.file_format = 'netcdf'
        etc_storm.scaling = [self.scaling, 1.0]
        etc_storm.window_type = 'custom'
        etc_storm.ramp_width = 2
        etc_storm.window = [-80, 27.5, -62.5, 45]
        etc_storm.write(self.rundata.surge_data.storm_file,
                                        file_format='data',
                                        dim_mapping={"t": "valid_time"},
                                        var_mapping={"pressure": "msl"},
                                        verbose=True)

        # Write out all data files
        super(ETCJob, self).write_data_objects()

def wrap_storm_coords(path):
    r"""Wrap storm coordinates and save to new file."""

    output_path = path.parent / f"{path.stem}_wrap.nc"
    util.wrap_coords(path, output_path=output_path,
                                dim_mapping={'t': 'valid_time'})
    return output_path

if __name__ == '__main__':

    base_path = (Path(os.environ['DATA_PATH']) / "ETC_NASA_SLCT").resolve()
    # storm_paths = [base_path / "DEC2012_0pt25.nc",
    #                          base_path / "DEC2012_1pt00.nc",
    #                          base_path / "DEC2012_1pt50.nc"]
    storm_paths = [base_path / "NOV2018_0pt25.nc",
                   base_path / "NOV2018_1pt00.nc",
                   base_path / "NOV2018_1pt50.nc"]

    jobs = []
    for sea_level in [0.0]:
        # for amr_max_levels in [2, 7]:
            # for scaling in [1.0, 1.1, 1.2]:
        for amr_max_levels in [7]:
            # for scaling in [1.2]:
            for scaling in [1.0]:
                for storm_path in storm_paths:
                    jobs.append(ETCJob(storm_path, 
                                       scaling=scaling, 
                                       sea_level=sea_level,
                                       levels=amr_max_levels))

    controller = batch.batch.BatchController(jobs)
    controller.wait = True
    controller.plot = True
    controller.parallel = True
    print(controller)
    controller.run()
