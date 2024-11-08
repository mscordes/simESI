import shutil
import numpy as np
from timeit import default_timer as timer
import os
from system import sys_info, update_system
from gmx import call_mdrun, modify_tc_grps, modify_ndx_grps
from ph import find_clusterpH
from pairs import find_pairs
from exchange import do_exchanges
from evaporate import remove_evaporated
from recenter import recenter_droplet, reset_gasVelocities
from coordinates import * 
from droplet_formation import atom_masses

def run_simulation(data_dir, args, pka_vals, top_order):
    """High level handling of the simulation and output writing including the master loop 
    for the 'stitching' method that simESI uses to faciliate proton exchanges.

    Args:
        data_dir (str): Pathname of the data subdir where high level information stored.
        args: argparse object of user defined args.  
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
             residue number, and values corresponding to its pKa as computed via PROPKA.
        top_order (list): Ordered list of residue names for .top files.
    
    Outputs:
        For every timestep, will save coordinate .gro file and write associated system 
        information like droplet composition and execution times to data subdir. Additionally, 
        prints this system information to terminal after each timestep to see simulation 
        progress at a glance.
    """
    print('Beginning production run.') 
    start_time = timer()

    # Continuation check and get initial prot/system charge for output
    if args.dir is None: 
        shutil.copyfile('nvt.gro', '0.gro')
        shutil.copyfile('prot.top', '0.top')
        system = sys_info('0.gro', top_order)
        proteins, _, _, _, _, _, _  = unpack_gro(f'0.gro')
    else: # If continuing, create topology file from .gro file
        top_from_coord(f'{args.step-1}.gro', f'{args.step-1}.top', args, top_order)
        system = sys_info(f'{args.step-1}.gro', top_order)
        proteins, _, _, _, _, _, _  = unpack_gro(f'{args.step-1}.gro')
    prot_charge = find_protCharge(proteins)

    # Calculate --water_cutoff based on protein mass, for ubiquitin (8.6 kDa) this is 30
    mass = sum(atom_masses[atom.element] for atom in system.protein_atoms)
    args.water_cutoff = args.water_cutoff if args.water_cutoff != -1 else round(mass*0.0035)

    # Write initial data
    def write_output(target_dir, input, output_fname):
        with open(os.path.join(target_dir, f'{output_fname}'), 'a+') as f:
            f.write(str(input) + '\n')
    for res in system.num_molecs:
        if res in ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
            write_output(data_dir, system.num_molecs[res], f'{res}.txt')
    write_output(data_dir, prot_charge, 'prot_charge.txt')
    write_output(data_dir, system.sys_charge, 'sys_charge.txt')

    # Each stitch is 4ps, so timecutoff in ns*250 = number of steps 
    steps = int(250.0*args.time)
    temperature = np.full(int(250.0*args.time), args.init_temp) #Droplet T

    # For GROMACS checkpoint file naming
    num_restarts = 1
    num_steps = 2000

    # Master for loop for running the simulation using the 'stitching' method
    for step in range(args.step, steps):
        start_time_step = timer()

        # Copy over coord/top files from previous step and update system
        shutil.copyfile(f'{step-1}.gro', f'{step}_preExchange.gro')
        shutil.copyfile(f'{step-1}.top', f'{step}.top')
        system = sys_info(f'{step}_preExchange.gro', top_order)

        # Sometimes pdb2gmx will miss disulfide bonds if bond slightly too long/short, so correct here
        system.atoms = fix_disulfides(system.residues, system.atoms)

        # Remove evaporated molecules
        system, poly_com, evap = remove_evaporated(system, args)
        if evap:
            # Update topology and system information
            modify_top(f'{step}.top', top_order, system.num_molecs)
            system = update_system(system, args, top_order, step, False)

        # We want to completely desolvate/kick off adducts now if waters below cutoff, so ramp to 450K
        if system.num_molecs['SOL'] < args.water_cutoff: 
            temperature = np.full_like(temperature, args.final_temp)      

        # Reset list of previous exchanges and skip_coords (skip coords prevent 'rattling' of proton between donor and acceptor)
        exchanges = []
        skip_coords = np.array([9999.9, 9999.9, 9999.9]) #foo coord as placeholder

        # Find localized pH/titratable sites, only do this on hop 0, or if residues have changed (ie. pairs flag)
        clusters, cluster_ph, cluster_waters = find_clusterpH(system.waterO_coords, system.res_dict['HHO'], system.res_dict['OHX'])
        tit_sites = {}
        tit_sites, system.res_dict = get_titSites(tit_sites, system, clusters, cluster_waters, True)

        # Flags to determine if new protein top required (prot) or if exchanges happen which only requires updating system info (pairs) 
        pairs = False
        prot = False

        # 5 proton hopping events per timestep (for Grotthuss diffusion of H3O+)
        for hop in range(5):
            # If no valid exchanges, pass through to MD portion
            if hop != 0 and pairs == False:
                break

            # Update localized pH/titratable sites if residues have changed (ie. pairs flag)
            if pairs:
                clusters, cluster_ph, cluster_waters = find_clusterpH(system.waterO_coords, system.res_dict['HHO'], system.res_dict['OHX'])
                tit_sites, system.res_dict = get_titSites(tit_sites, system, clusters, cluster_waters, prot)

            # Reset flags
            pairs = False
            prot = False

            # Find donor acceptor pairs
            accepted_pairs, skip_coords, pairs, prot, exchanges = \
                find_pairs(system, pka_vals, hop, cluster_ph, tit_sites, temperature[step], skip_coords, pairs, prot, exchanges)

            # Facilitate Exchanges
            system.atoms = \
                do_exchanges(accepted_pairs, system.atoms, system.residues, system.proteins, system.positions, top_order, pairs, prot, system)

            # Create new coord/top files if amino acid protonation states changed and updates system info
            if pairs or prot:
                system = update_system(system, args, top_order, step, prot)

        # Recenters droplet if it drifts towards edge of box
        recenter = False
        if np.linalg.norm(poly_com - np.array([vector/2 for vector in system.box_vectors])) > 5 and args.step != step:
            system.atoms = recenter_droplet(system.atoms, system.residues, system.box_vectors, poly_com, system.res_dict)
            recenter = True

        # Gas velocities can become distorted due to stitching, correct here if atm present
        gas_correct = False
        if args.atm:
            system.atoms, gas_correct = reset_gasVelocities(system, gas_correct)

        # Post exchange structure
        write_gro(f'{step}_postExchange.gro', system.atoms, system.box_vectors)
        end_time_exchange = timer()

        # Prep mdp file for MD, tc-grps will break unless the specified residue name is present
        modify_tc_grps('prodrun.mdp', temperature[step], system.num_molecs, args)
        
        # Start MD
        if exchanges or recenter or gas_correct or evap or step == args.step or step%50 == 0:
            # Run file
            modify_ndx_grps(f'{step}_postExchange', f'{step}', system.num_molecs, args)
            command = f'gmx grompp -f prodrun.mdp -c {step}_postExchange.gro -p {step}.top -n {step}.ndx -o {step}.tpr -maxwarn 200'
            auto_gmx_input(command, None, args)
            grompp_time = timer()
            
            # MD run
            call_mdrun(f'{step}', args)
            end_time_MD = timer()
            num_restarts = 1 #Checkpoint naming stuff
            num_steps = 2000

            # Cleanup outputs
            if not args.save:
                os.remove(f'{step}.edr')
                os.remove(f'{step}.log')

        # If no exchanges, we can simply modify the run file from previous step rather than calling grompp again
        else:
            num_restarts += 1
            num_steps += 2000
            shutil.copyfile(f'{step-1}.top', f'{step}.top')
            shutil.copyfile(f'{step-1}.cpt', f'{step}.cpt')
            shutil.copyfile(f'{step-1}.ndx', f'{step}.ndx')

            # Modify run file to extend time
            command = f'gmx convert-tpr -s {step-1}.tpr -o {step}.tpr -nsteps {num_steps}'
            auto_gmx_input(command, None, args)
            grompp_time = timer()

            # MD run
            if args.hpc: #HPC cluster, edit command if pulling different resources
                command = f'gmx mdrun -ntmpi 1 -nb gpu -cpi {step} -deffnm {step} -noappend -cpo {step}'      
            elif args.gpu: #GPU accelerated (not HPC)
                command = f'gmx mdrun -nb gpu -cpi {step} -deffnm {step} -noappend -cpo {step}'             
            else: #CPU only
                command = f'gmx mdrun -cpi {step} -deffnm {step} -noappend -cpo {step}'
            auto_gmx_input(command, None, args)

            # Checkpoint naming
            os.replace(f'{step}.part{str(num_restarts).zfill(4)}.gro', f'{step}.gro') 
            try:
                os.replace(f'{step}.part{str(num_restarts).zfill(4)}.xtc', f'{step}.xtc')
            except FileNotFoundError:
                pass
            end_time_MD = timer()

            # Cleanup outputs
            if not args.save:
                os.remove(f'{step}.part{str(num_restarts).zfill(4)}.edr')
                os.remove(f'{step}.part{str(num_restarts).zfill(4)}.log')

        # Final cleanup outputs
        if not args.save:
            if os.path.exists(f'{step}.gro'):
                os.remove(f'{step}_preExchange.gro')
                os.remove(f'{step}_postExchange.gro')
            else:
                print(f'Run failed at MD run during step {step}.')
                print('Exchanges at failure were...')
                [print(exchange) for exchange in exchanges]
                exit(1)
            if not step%10 == 0 and os.path.exists(f'{step}.xtc'):
                os.remove(f'{step}.xtc')
            if step > 2:
                if os.path.exists(f'{step-2}_prev.cpt'):
                    os.remove(f'{step-2}_prev.cpt')
                try:
                    os.remove(f'{step-2}.cpt')
                    os.remove(f'{step-2}.tpr')
                    os.remove(f'{step-2}.ndx')
                    os.remove(f'{step-2}.top')
                except FileNotFoundError:
                    pass

        # Outputs
        end_time = timer()
        exchange_time = end_time_exchange - start_time_step
        MD_prep_time = grompp_time - end_time_exchange
        MD_time = end_time_MD - grompp_time
        tot_time = (end_time - start_time)/3600
        simulation_time = float(step)*0.004
        prot_charge = find_protCharge(proteins)
        sys_charge = prot_charge + system.num_molecs['HHO'] - system.num_molecs['OHX'] + system.num_molecs['NXH'] - system.num_molecs['ATX']

        # Store clustering information
        with open(os.path.join(data_dir, 'cluster.txt'), 'a+') as f:
            f.write(f'{cluster_waters} {cluster_ph} {step}' + '\n')

        # Write data to associated files
        with open(os.path.join(data_dir, 'exchanges.txt'), 'a+') as f:
            if exchanges:
                for exchange in exchanges:
                    f.write(f'{step} {exchange.hop} {exchange.donor.atom_name} {exchange.donor.res_id} '
                            f'{exchange.acceptor.atom_name} {exchange.acceptor.res_id} {exchange.energy} '
                            f'{exchange.cluster} {exchange.near_waters}\n')
        for res in system.num_molecs:
            if res in ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
                write_output(data_dir, system.num_molecs[res], f'{res}.txt')

        write_output(data_dir, sys_charge, 'sys_charge.txt')
        write_output(data_dir, system.prot_charge, 'prot_charge.txt')
        write_output(data_dir, exchange_time, 'exchange_time.txt')
        write_output(data_dir, MD_time, 'md_time.txt')
        write_output(data_dir, MD_prep_time, 'grompp_time.txt')
        write_output(data_dir, tot_time, 'tot_time.txt')
        write_output(data_dir, temperature[step], 'temperature.txt')

        # Print outputs to terminal
        print(f'\n\n\n------ TIMESTEP {step} COMPLETED ------')
        print('\n--------- EXECUTION TIME ----------')
        print(f'Current working dir:     {data_dir}')
        print(f'Exchange calculation:    {"{:.2f}".format(exchange_time)} s')
        print(f'MD preparation:          {"{:.2f}".format(MD_prep_time)} s')
        print(f'MD simulation:           {"{:.2f}".format(MD_time)} s')
        print(f'Total timestep time:     {"{:.2f}".format(end_time - start_time_step)} s')
        print(f'Total time elapsed:      {"{:.2f}".format(tot_time)} hrs')
        print('----------- SYSTEM INFO -----------')
        print(f'Time simulated:          {"{:.3f}".format(simulation_time)} ns')
        print(f'System net charge:       {system.sys_charge}')
        print(f'Protein net charge:      {system.prot_charge}')
        print(f'Number of H3O+:          {system.num_molecs["HHO"]}')
        print(f'Number of OH-:           {system.num_molecs["OHX"]}')
        print(f'Number of acetates:      {system.num_molecs["ATX"]}')
        print(f'Number of acetic acids:  {system.num_molecs["AHX"]}')
        print(f'Number of NH3:           {system.num_molecs["NXX"]}')
        print(f'Number of NH4:           {system.num_molecs["NXH"]}')
        print(f'Number of waters:        {system.num_molecs["SOL"]}')
        print(f'System temperature:      {"{:.2f}".format(temperature[step])} K')
        print('\n----------- EXCHANGES -----------')
        if exchanges:
            for exchange in exchanges:
                print(f'From {exchange.donor.atom_name} (residue {exchange.donor.res_id}) to {exchange.acceptor.atom_name} '
                      f'(residue {exchange.acceptor.res_id}) during hop {exchange.hop}.')
        print('---------------------------------')

        #If completely desolvated (ie. 0 waters), end run unless --end is 'cutoff'. 
        if args.end == 'delsolvated' and system.num_molecs["SOL"] == 0:
            print('Protein successfully desolvated.')
            print(f'Final protein charge is {prot_charge}.')
            exit(0)

    print(f'Time limit of {args.time} ns reached.')
    print(f'Final protein charge is {prot_charge}.')
    exit(0)