import argparse

def get_args():
    """ 
    Parses commmand line args.

    Outputs:
        args (argparse object). 
    """
    parser = argparse.ArgumentParser(
                    prog='simESI',
                    description='simESI (simulations of ESI) is a package of scripts written in entirely ' + 
                    'in python utilizing GROMACS for simulating electrospray ionization (ESI) of proteins ' +
                    'in ammonium acetate containing droplets to form protonated or deprotonated protein ions.',
                    epilog='Version 1.0')

    parser.add_argument('--pdb', type=str, required=True, \
                        help='pdb filename in simESI-main/input_files. Example of a default included in simESI '+
                         'is "ubq.pdb" (PDB:1UBQ).')

    parser.add_argument('--esi_mode', type=str, choices=['pos', 'neg'], default='pos', \
                        help='Simulate positive-ion mode ("pos") or negative-ion mode ("neg"). Default of "pos".')

    parser.add_argument('--amace_conc', type=float, default=0.25, \
                        help='(Float) If "amace" selected for ion_seeding, chooses initial ammonium acetate concentration. Default of 0.25.')

    parser.add_argument('--droplet_size', type=float, default=None, 
                        help='(Float) Radius of droplet in nm. Defaults to 1.8x protein radius (assuming 1.22 g/cm^3 protein density).')

    parser.add_argument('--end', type=str, choices=['desolvated', 'cutoff'], default='desolvated', \
                        help='Choose when to end the simulation, options are "desolvated" which ends simulation if number of waters == 0 or ' +
                            '"cutoff" once time cutoff as defined by --time reached. Default of "desolvated".')

    parser.add_argument('--time', type=float, default=25, \
                        help='(Float) Cutoff in ns of when to end the simulation. Default of 25 ns.')

    parser.add_argument('--water_cutoff', type=int, default=-1, \
                        help='(Int) Cutoff for # of waters when protein considered nearly desolvated, ramp temp at this point to fully desolvate.' + 
                        'By default, is set corresponding to protein mass, for ubquitin this (8.6 kDa) is ~30.')

    parser.add_argument('--atm', type=str, choices=['yes', 'no'], default='yes', \
                        help='Choose to simulate droplet in ambient conditions with atmosphere if "yes", else "no" for vacuum.')

    parser.add_argument('--init_temp', type=float, default=370, \
                        help='(Float) Initial droplet temp in K. Defualt of 370 K.')

    parser.add_argument('--final_temp', type=float, default=450, \
                        help='(Float) Final droplet temp to ramp to once water_cutoff reached in K. Defualt of 450, setting equal to ' +
                            '--init_temp removes any temperature ramp (i.e., 370 if using default --init_temp).')

    parser.add_argument('--gpu', type=str, choices=['yes', 'no'], default='yes', \
                        help='Choose to use GPU acceleration "yes", if CPU only then, "no". CPU only not recommended for ' +
                        'proteins larger than ubiquitin.')

    parser.add_argument('--hpc', type=str, choices=['yes', 'no'], default='no', \
                        help='Options for using (my) HPC cluster. Assumes 1 node with 18 ranks and GPU acceleration. If pulling ' +
                        'different resources, modify the gmx command in line 18 in gmx.py and line 182 in simulation.py.')

    parser.add_argument('--verbose', type=str, choices=['yes', 'no'], default='no', \
                        help='Chooses whether or not to hide gmx outputs in terminal.')

    parser.add_argument('--save', type=str, choices=['yes', 'no'], default='no', \
                            help='Choose whether to save different output files associated with gmx mdrun, ie. .log, .edr files, etc. ' +
                            'These can take up a lot of memory!')

    parser.add_argument('--dir', type=str, default=None, \
                        help='(Str) Continue run from a given trial dir. I.e., if ubq_1 is trial dir, set --dir to ubq_1. Also see --step. \
                                WARNING: Must call --pdb again with the inputted .pdb file used to start.')
    
    parser.add_argument('--step', type=int, default=1, \
                        help='(Int) Step to continue from, i.e., the step at failure.')

    parser.add_argument('--equil', type=str, choices=['yes', 'no'], default='yes', \
                        help='Skips droplet formation/equilibriation. simESI generally sets this as needed, so ' +
                        'shouldnt need to set unless debugging.')

    parser.add_argument('--pka_pdb', type=str, default=None, \
                        help='(Str) Use seperate .pdb file for generating pKa vals.')

    args = parser.parse_args()

    # Some args need to checked frequently so convert to bool
    args.atm = args.atm == 'yes'
    args.gpu = args.gpu == 'yes'
    args.hpc = args.hpc == 'yes'
    args.verbose = args.verbose == 'yes'
    args.save = args.save == 'yes'
    args.equil = args.equil == 'yes'

    # If restarting run, don't want to redo equilibriation
    if args.dir is not None:
        args.equil = False
    return args