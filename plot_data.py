#!/usr/bin/env python

from pathlib import Path
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def plot_fields(path, dates, movie=True, force=False):

    data = xr.open_dataset(path)
    stride = 2
    x = np.mod(data.longitude - 360, 180)[::stride]
    y = data.latitude[::stride]
    t_range = dates
    subset = data.sel(valid_time=slice(*t_range))
    t = (subset.valid_time - t_range[0]) / np.timedelta64(1, 's')
    
    out_path = Path() / "storm_plots" / path.stem
    out_path.mkdir(exist_ok=True)

    xlimits = (80, 130)
    ylimits = (10, 60)
    W_limits = [0, 30]
    P_limits = [900,1000]
    for (n, time) in enumerate(t):
        
        image_path = out_path / f"frame_W{str(n).zfill(5)}.png"
        if force or not image_path.exists():
            wind = [subset.u10, subset.v10]
            wind_speed = np.sqrt(subset.u10[n, ::stride, ::stride]**2 + subset.v10[n, ::stride, ::stride]**2)
        
            fig_W, ax_W = plt.subplots()
            plot = ax_W.pcolor(x, y, wind_speed, vmin=W_limits[0], vmax=W_limits[1])
            cbar = fig_W.colorbar(plot, label=r"$|W|$ ($m/s$)")
            ax_W.set_title(f"Wind Speed - {subset.valid_time[n].values}")
            ax_W.set_xlim(xlimits)
            ax_W.set_ylim(ylimits)
            fig_W.savefig(image_path)
            del(fig_W)

        image_path = out_path / f"frame_P{str(n).zfill(5)}.png"
        if force or not image_path.exists():
            fig_P, ax_P = plt.subplots()
            # pressure = subset.msl[n, :, :]
            pressure = subset.sp[n, ::stride, ::stride] / 1e2
            plot = ax_P.pcolor(x, y, pressure, vmin=P_limits[0], vmax=P_limits[1])
            cbar = fig_P.colorbar(plot, label=r"$P$ ($hPa$)")
            ax_P.set_xlim(xlimits)
            ax_P.set_ylim(ylimits)
            ax_P.set_title(f"Surface Pressure - {subset.valid_time[n].values}")
            fig_P.savefig(image_path)
            del(fig_P)

        print(f"Done with frame {n + 1}/{t.shape[0]}.")

    # Create movie
    if movie:
        for field in ['W', 'P']:
            frames_per_second = 6
            frames_per_second_mov = 24
            image_paths = out_path / f"frame_{field}%05d.png"
            output_file = out_path / f"_{field}.mp4"
            cmd = (f"ffmpeg -r {frames_per_second} -i {image_paths} " +
                   f"-q:a 0 -q:v 0 -vcodec mpeg4 -vb 20M -r " +
                   f"{frames_per_second_mov} {output_file}")
            print(cmd)
            subprocess.call(cmd, shell=True)
        


if __name__ == '__main__':
    # path = Path(os.environ['DATA_PATH']) / "ETC_NASA_SLCT" / "old_data" / "f166d10549b1da216d3d9a1a3d9f6af2.nc"
    # file_names = ["DEC2012_1pt50.nc", "DEC2012_1pt00.nc", "DEC2012_0pt25.nc"]
    dates = (np.datetime64("2012-12-25"), np.datetime64("2012-12-30"))
    # file_names = ["NOV2018_0pt25.nc", "NOV2018_1pt00.nc", "NOV2018_1pt50.nc"]
    # dates = (np.datetime64("2018-11-01"), np.datetime64("2018-11-30"))
    # for file_name in file_names:
    #     plot_fields(Path(os.environ['DATA_PATH']) / "ETC_NASA_SLCT" / file_name,
    #                 dates)
    plot_fields(Path() / "output.nc", dates)
