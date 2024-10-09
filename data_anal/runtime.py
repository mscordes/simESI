#! /usr/bin/python
'''
When given a trial directory in the output_files subdirectory (ie, with defaults something like ubq_1),
will generate a plot showing the different contributions to simulation time. This includes contributions 
from the exchange calculation, MD prep time (ie, the time it takes for gmx grompp to make a run file),
and the actual MD runs per every timestep. Additionally, this will plot a sum of the three aformentioned
contributions as well as a cumulative time plot.

Example call: python sysinfo.py --dir ubq_1
'''
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
import numpy as np

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True) #Name of dir in simESI_main/output_files
parser.add_argument('--save', type=str, choices=['yes', 'no'], default='no') #Choose whether to auto save plot
args = parser.parse_args()

dirpath = os.path.join('../output_files', args.dir, 'data')
def openfile(filename):
    fpath = os.path.join(dirpath, filename)
    with open(f'{fpath}', 'r') as f:
        return np.array([float(line) for line in f.readlines()])

# Different time contributions
exchange = openfile('exchange_time.txt') #Exchange calculations
grompp = openfile('grompp_time.txt') #gmx grompp calls
simulation = openfile('md_time.txt') #Actual MD simulation
total = exchange + grompp + simulation
runtime = np.cumsum(total) / 3600

# Begin plotting
fig = plt.figure()
fig.set_size_inches(16, 9)
gs = gridspec.GridSpec(1, 2)
gss = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[0])
plt.rcParams["font.size"] = 15
plt.rcParams['lines.linewidth'] = 0.75
xmax = len(total)
ymax = np.max(total)

def plot_data(ax, data, label):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(f'{label}')
    ax.plot(data, color='black')
    return ax

# Exchange calculations
ax0 = fig.add_subplot(gss[0])
ax0 = plot_data(ax0, exchange, 'Exchange (s)')
ax0.get_xaxis().set_ticks([])

# gmx grompp
ax1 = plt.subplot(gss[1])
ax1 = plot_data(ax1, grompp, 'grompp (s)')
ax1.get_xaxis().set_ticks([])

# MD simulation time
ax2 = plt.subplot(gss[2])
ax2 = plot_data(ax2, simulation, 'MD (s)')
ax2.get_xaxis().set_ticks([])

# Total time per timestep
ax3 = plt.subplot(gss[3])
ax3 = plot_data(ax3, total, 'Total (s)')
ax3.set_xlabel('Timestep')

# Total runtime
ax4 = plt.subplot(gs[1])
ax4 = plot_data(ax4, runtime, 'Total Runtime (hr)')
ax4.set_ylim(0, np.max(runtime))
ax4.set_xlabel('Timestep')

fig.subplots_adjust(hspace=0.15)
fig.align_labels()

if args.save == 'yes':
    plt.draw()
    plt.savefig(f'runtime_{args.dir}.png', dpi=300, bbox_inches='tight')
elif args.save == 'no':
    plt.show()