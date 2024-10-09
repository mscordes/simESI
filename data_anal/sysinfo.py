#! /usr/bin/python
'''
When given a trial directory in the output_files subdirectory (ie, with defaults something like ubq_1),
will generate a stacked plot displaying various changes in droplet composition as the simulation progresses.
This includes number of each molecule in the droplet, protein charge, and ratio of droplet charge to Rayleigh
limit. Requires protein mass be inputted to the --mass arg in kDa (ie., 8.56 for ubiquitin).

Example call: python sysinfo.py --dir ubq_1 --mass 8.56
'''
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import math
import argparse
import numpy as np

# User inputs
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True) #Name of dir in simESI_main/output_files (0_1ubq as example)
parser.add_argument('--mass', type=float, required=True) #Mass of protein in kDa
parser.add_argument('--save', type=str, choices=['yes', 'no'], default='no') #Choose whether to auto save plot
args = parser.parse_args()

# Constants, unless noted all numerical calcs are done in SI units
dirpath = os.path.join('../output_files', args.dir, 'data')
Na = 6.0221408*10**23 #Avagrados No.
e = 1.60217663*10**-19 #e- charge (C)
eo = 8.854187*10**-12 #Vacuum permitivity (F/m)
gamma = 0.0727 #Water surface tension (N m^-1)
rl_const = (8*math.pi/e)*((3*eo*gamma)/(4*math.pi))**(1/2) #Simplified constant for Rayleigh calc's
water_density = 1000 #kg/m^3
prot_density = 1220 #kg/m^3
H2O_vol = 0.01801/(Na*water_density) #Vol of single H2O molecule (m^3/molec)

# Rayeigh limit of a given radius
def rayleigh(num_water):
    Vprot = (args.mass/Na)/prot_density #protein volume
    return rl_const*(Vprot+H2O_vol*num_water)**(1/2)

def openfile(filename):
    fpath = os.path.join(dirpath, filename)
    with open(f'{fpath}', 'r') as f:
        return np.array([float(line) for line in f.readlines()])

# Get droplet composition and simulation time
composition = {}
for molec in ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
    composition[molec] = openfile(f'{molec}.txt')
time = [index*0.004 for index in range(len(composition['SOL']))]

# Get protein and system charge
composition['PR'] = openfile('prot_charge.txt')
composition['SYS'] = composition['PR'] + composition['HHO'] + composition['NXH'] - composition['OHX'] - composition ['ATX']

# Get ratio of system charge to Rayleigh limit
composition['RL'] = rayleigh(composition['SOL'])
composition['QZR'] = composition['SYS'] / composition['RL']

print(f'Final protein charge is {composition["PR"][-1]}.')
print(f'Final system charge (including adducts) is {composition["SYS"][-1]}.')

# Begin plotting
fig = plt.figure()
fig.set_size_inches(8, 7)
plt.rcParams["font.size"] = 15
plt.rcParams['lines.linewidth'] = 0.75
gs = gridspec.GridSpec(5, 1)

# Waters
ax0 = plt.subplot(gs[0])
ax0.set_xlim(0, time[-1])
ax0.set_ylim(0, np.max(composition['SOL']))
ax0.set_ylabel('H₂O')
ax0.get_xaxis().set_ticks([])
ax0.plot(time, composition['SOL'], color='black')

# Acetate/acetic acid
ax1 = plt.subplot(gs[1])
ax1.set_xlim(0, time[-1])
buff_max = max(np.max(composition['ATX']), np.max(composition['NXH']))
ax1.set_ylim(0, buff_max)
ax1.set_ylabel('Ac')
ax1.get_xaxis().set_ticks([])
ax1.plot(time, composition['AHX'], label='CH₃CO₂H', color='black')
ax1.plot(time, composition['ATX'], label='CH₃CO₂⁻', color='red')
ax1.legend(loc='upper right', frameon=False)

# Ammonium/ammonia
ax2 = plt.subplot(gs[2])
ax2.set_xlim(0, time[-1])
ax2.set_ylim(0, buff_max)
ax2.set_ylabel('NHₓ')
ax2.get_xaxis().set_ticks([])
ax2.plot(time, composition['NXH'],  label='NH₄⁺', color='black')
ax2.plot(time, composition['NXX'],  label='NH₃', color='red')
ax2.legend(loc='upper right', frameon=False)

# Protein charge
ax3 = plt.subplot(gs[3])
ax3.set_xlim(0, time[-1])
ax3.set_ylabel('Protein z')
ax3.get_xaxis().set_ticks([])
ax3.plot(time, composition['PR'], color='black')

# Ratio of droplet charge to Rayleigh ratio
ax4 = plt.subplot(gs[4])
ax4.set_xlim(0, time[-1])
ax4.set_ylim(0, max(1.2, np.max(composition['QZR'])))
ax4.set_xlabel('Time (ns)')
ax4.set_ylabel('z/zᵣ')
ax4.hlines(y=1.0, xmin=0, xmax=time[-1], linestyle='dashed', color='black', label='Rayleigh Limit')
ax4.plot(time, composition['QZR'], color='black')
ax4.legend(loc='lower right', frameon=False)

fig.subplots_adjust(hspace=0.15)
fig.align_labels()

if args.save == 'yes':
    plt.draw()
    plt.savefig(f'sysInfo_{args.dir}.png', dpi=300, bbox_inches='tight')
elif args.save == 'no':
    plt.show()