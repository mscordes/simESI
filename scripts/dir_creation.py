import os
import shutil

def create_dirs(args):
    """Creates unique, numbered trial subdirectory in the simESI/output_files dir and loads it 
    with required files. Example would be 'simESI/output_files/ubq_1' if running with default 
    ubq.pdb with no other trial runs. 'ubq_1' itself has two subdir's including, 'ubq_1/data' 
    containing high level information to expedite data analysis, and 'ubq_1/simulation' containing
    files related to the MD simulation, i.e., coordinate .gro files.

    Args:
        args: argparse object of user line args.

    Args:
        data_dir (str): Pathname of the data subdir.
    """
    simESI_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    output_dir = os.path.join(simESI_dir, 'output_files')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    trial = 1
    if not args.dir:
        while True:
            trial_dir = os.path.join(output_dir, f'{args.pdb[0:-4]}_{trial}')
            if os.path.isdir(trial_dir):
                trial += 1
            else:
                os.mkdir(trial_dir)

                # Make data dir to store high level info about simulation.
                data_dir = os.path.join(trial_dir, 'data')
                os.mkdir(data_dir)

                # Actual simulation dir.
                simulation_dir = os.path.join(trial_dir, 'simulation')
                os.mkdir(simulation_dir)
                os.chdir(simulation_dir)

                # Load required files into simulation dir like FF, coord files, etc.
                shutil.copyfile(os.path.join(simESI_dir, 'input_files', args.pdb), args.pdb)
                shutil.copytree(os.path.join(simESI_dir, 'share', 'charmm36.ff'), 'charmm36.ff')
                topology_dir = os.path.join(simESI_dir, 'share', 'top_files')
                for file in os.listdir(topology_dir):
                    shutil.copyfile(os.path.join(topology_dir, file), file)
                mdp_dir = os.path.join(simESI_dir, 'share', 'mdp_files')
                for file in os.listdir(mdp_dir):
                    shutil.copyfile(os.path.join(mdp_dir, file), file)            

                # Copy over inputted .pdb
                shutil.copyfile(os.path.join(simESI_dir, 'input_files', f'{args.pdb}'), f'{args.pdb}')              
                if args.pka_pdb is not None:
                    shutil.copyfile(os.path.join(simESI_dir, 'input_files', args.pka_pdb), args.pka_pdb)
                break

    # Choose to continue previous run
    elif args.dir:
        data_dir = os.path.join(simESI_dir, 'output_files', args.dir, 'data')
        sim_dir = os.path.join(simESI_dir, 'output_files', args.dir, 'simulation')
        os.chdir(sim_dir)

        # Copy over required files (if not in simulation dir)
        if os.path.exists('charmm36.ff') == False:
            shutil.copytree(os.path.join(simESI_dir, 'share', 'charmm36.ff'), 'charmm36.ff')
        if os.path.exists(f'{args.pdb}') == False:
            shutil.copyfile(os.path.join(simESI_dir, 'input_files', args.pdb), args.pdb)
        topology_dir = os.path.join(simESI_dir, 'share', 'top_files')
        for file in os.listdir(topology_dir):
            shutil.copyfile(os.path.join(topology_dir, file), file)
        mdp_dir = os.path.join(simESI_dir, 'share', 'mdp_files')
        for file in os.listdir(mdp_dir):
            shutil.copyfile(os.path.join(mdp_dir, file), file)     
        if args.pka_pdb is not None:
            shutil.copyfile(os.path.join(simESI_dir, 'input_files', args.pka_pdb), args.pka_pdb)
    return data_dir