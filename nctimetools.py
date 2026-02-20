#!/usr/bin/env python
"""
NetCDF Time Manipulation and Plotting Module

This module provides functionality to:
- Adjust time coordinates by adding/subtracting time deltas
- Resample time coordinates to change spacing
- Plot variables over time
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse
from datetime import timedelta
import sys

def resample_time(input_path, multiplier, output_path=None):
    """Resample time coordinates by a multiplier without interpolation."""
    data = xr.open_dataset(input_path)

    # Store original time encoding
    time_encoding = data.valid_time.encoding

    # Get time values and reference point
    times = data.valid_time.values
    t0 = times[0]
    time_deltas = times - t0

    # Scale time differences
    new_time_deltas = time_deltas * multiplier
    new_times = t0 + new_time_deltas

    # Update time coordinate
    data = data.assign_coords(valid_time=new_times)

    # Restore original encoding
    data.valid_time.encoding = time_encoding

    # Save output
    if output_path is None:
        output_path = input_path

    data.to_netcdf(output_path)
    data.close()
    print(f"Time spacing changed by factor {multiplier}, saved to {output_path}")


def adjust_time(input_path, delta, output_path=None):
    """Adjust time coordinates by adding/subtracting a time delta."""
    data = xr.open_dataset(input_path)

    # Store original time encoding
    time_encoding = data.valid_time.encoding

    # Convert timedelta to nanoseconds
    delta_ns = int(delta.total_seconds() * 1e9)
    new_time = data.valid_time + np.timedelta64(delta_ns, 'ns')

    # Update time coordinate
    data = data.assign_coords(valid_time=new_time)

    # Restore original encoding
    data.valid_time.encoding = time_encoding

    # Save output
    if output_path is None:
        output_path = input_path

    data.to_netcdf(output_path)
    data.close()
    print(f"Time adjusted by {delta}, saved to {output_path}")


def plot_variable(input_path, variable_name, output_plot=None):
    """Plot a variable over time."""
    try:
        data = xr.open_dataset(input_path)

        # Check if variable exists
        if variable_name not in data:
            print(f"Error: Variable '{variable_name}' not found in dataset.")
            print(f"Available variables: {list(data.variables)}")
            sys.exit(1)

        # Create plot
        plt.figure(figsize=(10, 6))
        data[variable_name].plot(x='valid_time')
        plt.title(f"{variable_name} over time")
        plt.ylabel(variable_name)
        plt.xlabel("Time")
        plt.grid(True, alpha=0.3)

        if output_plot:
            plt.savefig(output_plot, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_plot}")
        else:
            plt.show()

        data.close()

    except Exception as e:
        print(f"Error plotting: {e}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NetCDF Time Manipulation and Plotting Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adjust time by 6 hours
  python nc_tool.py adjust input.nc output.nc --hours 6

  # Change time spacing to half (e.g., 6h -> 3h)
  python nc_tool.py resample input.nc output.nc --multiplier 0.5

  # Plot temperature variable
  python nc_tool.py plot input.nc temperature --output plot.png

  # Chain operations: adjust time then plot
  python nc_tool.py adjust input.nc temp.nc --hours 3 && \
                         python nc_tool.py plot temp.nc temperature && \
                         rm temp.nc
""")

    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Adjust time parser
    adjust_parser = subparsers.add_parser('adjust', help='Adjust time by adding/subtracting a delta')
    adjust_parser.add_argument('input', help='Input netCDF file')
    adjust_parser.add_argument('output', nargs='?', default=None, help='Output netCDF file (optional)')
    adjust_parser.add_argument('--days', type=float, default=0, help='Days to add/subtract')
    adjust_parser.add_argument('--hours', type=float, default=0, help='Hours to add/subtract')
    adjust_parser.add_argument('--minutes', type=float, default=0, help='Minutes to add/subtract')

    # Resample time parser
    resample_parser = subparsers.add_parser('resample', help='Change time spacing by multiplier')
    resample_parser.add_argument('input', help='Input netCDF file')
    resample_parser.add_argument('output', nargs='?', default=None, help='Output netCDF file (optional)')
    resample_parser.add_argument('--multiplier', type=float, default=1.0, 
    			       help='Time spacing multiplier (e.g., 0.5 for half, 2.0 for double)')

    # Plot parser
    plot_parser = subparsers.add_parser('plot', help='Plot variable over time')
    plot_parser.add_argument('input', help='Input netCDF file')
    plot_parser.add_argument('variable', help='Variable name to plot')
    plot_parser.add_argument('--output', '-o', help='Output plot file (optional, shows plot if not specified)')

    args = parser.parse_args()

    try:
        if args.command == 'adjust':
            delta = timedelta(days=args.days, hours=args.hours, minutes=args.minutes)
            adjust_time(args.input, delta, args.output)

        elif args.command == 'resample':
            resample_time(args.input, args.multiplier, args.output)

        elif args.command == 'plot':
            plot_variable(args.input, args.variable, args.output)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
