from arguments import get_args
from dir_creation import create_dirs
from equilibriation import equilibriate
from simulation import run_simulation

def main():
    """ 
    Get user defined args, create trial dirs, form droplet & equilbriate, 
    then run the simulation.
    """
    args = get_args()
    data_dir = create_dirs(args)
    pka_vals, top_order = equilibriate(args)
    run_simulation(data_dir, args, pka_vals, top_order)