#!/usr/bin/env python

from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import cf_xarray.units
import pint_xarray
from pint_xarray import unit_registry as ureg

xr.set_options(display_expand_data=False)

def plot_fields(path):

    data = xr.open_dataset(path, engine="netcdf4", decode_cf=True)
    x = np.mod(data.longitude - 360, 180)
    y = data.latitude
    t_range = (np.datetime64("2012-12-01"), np.datetime64("2012-12-10"))
    subset = data.sel(valid_time=slice(*t_range))
    t = (subset.valid_time - t_range[0]) / np.timedelta64(1, 's')
    n = 100
    wind = [subset.u10, subset.v10]
    wind_speed = np.sqrt(subset.u10[n, :, :]**2 + subset.v10[n, :, :]**2).pint.quantify(data.u10.units)
    msl_pressure = subset.msl[n, :, :].pint.quantify(data.msl.units)
    pressure = subset.sp[n, :, :].pint.quantify(data.sp.units)

    fig, ax = plt.subplots()
    plot = ax.pcolor(x, y, wind_speed)
    cbar = fig.colorbar(plot, label=r"$|W|$ (${:~L}$)".format(wind_speed.pint.units))
    ax.set_xlim((80, 130))
    ax.set_ylim((10, 60))
    ax.set_title(f"Wind Speed - {subset.valid_time[n].values}")

    fig, ax = plt.subplots()
    plot = ax.pcolor(x, y, pressure)
    cbar = fig.colorbar(plot, label=r"$P$ (${:~L}$)".format(pressure.pint.units))
    ax.set_xlim((80, 130))
    ax.set_ylim((10, 60))
    ax.set_title(f"Surface Pressure - {subset.valid_time[n].values}")

if __name__ == '__main__':
    path = Path(os.environ['DATA_PATH']) / "ETC_NASA_SLCT" / "f166d10549b1da216d3d9a1a3d9f6af2.nc"
    plot_fields(path)
    plt.show()