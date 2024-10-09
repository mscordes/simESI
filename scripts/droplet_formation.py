import numpy as np
import subprocess
import shutil
import math
from classes import Atom, Residue
from coordinates import *

# Physical Constants
Na = 6.022*10**23 # Avagadros no.
couloumb = 1.602176634*10**-19 # C per e- 
gas_const = 8.2057366*10**-5 # Gas constant (m^3*atm*K^-1*mol^-1)
prot_density = 1.22 #g/cm^3
atom_masses = {'O':15.99, 'C': 12.000, 'N':14.007, 'H':1.0078, 'S':32.065, 'M':0.0}


def form_atmosphere(atoms, positions, box_vectors, top_order):
    """
    Seeds atmosphere of 79% N2 and 21% O2 at standard temperature and pressure
    around droplet. 

    Args:
        atoms (list): Ordered list of Atoms objects.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        box_vectors (list): 3D coordinates defining box size.
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        Complete coordinate file of protein in drift tube with atm (droplet.gro). 
    """ 

    # Volume in which to seed gas
    box_vol = np.prod(box_vectors)

    # Number of individual gas molecules
    pressure = 1 #atm 
    temp = 300 #K 
    n = pressure*Na / (temp*gas_const) #number density (molec/m^3)
    n *= 10**-27 #convert to molec/nm^3
    num_gas = round(box_vol*n)

    # Approximate volume of protein for displacement
    prot_mass = sum(atom_masses[atom.element] for atom in atoms)
    prot_vol = ((prot_mass/Na) / prot_density) * (10**21) #nm^3
    displaced_gas = round(prot_vol*n)
    num_gas -= displaced_gas

    """Create evenly spaced grid of initial gas positions then perturb. 
    Start slightly in from box edge to prevent splitting of gas molecs."""

    #nm, ~2 N^2 widths, closest gas molecs can be
    min_space = 0.6 

    # Because need ints, rounding will lead to less than gas_num being seeded so scale up until threshold met
    scale = 1.0
    num_points = round(num_gas**(1/3))
    while True:
        temp_points = round(num_points*scale)
        if temp_points**3 > num_gas:
            num_points = temp_points
            break
        else:
            scale += 0.1

    # Mesh-grid
    grid_points = np.linspace(0.3, box_vectors[0]-0.3, num_points)  
    X, Y, Z = np.meshgrid(grid_points, grid_points, grid_points, indexing='ij')
    locs = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Perturb points
    init_dist = np.linalg.norm(locs[1] - locs[0])
    max_movement = (init_dist - min_space)/2
    noise = np.random.uniform(low=-max_movement,
                                high=max_movement,
                                size=(len(locs), 3))
    locs += noise

    # Delete extra points from scaling above
    extra = (num_points**3) - num_gas
    toDelete = np.random.randint(0, len(locs)-1, extra)
    locs = np.delete(locs, toDelete, 0)

    # Delete gas molecules that overlap with protein
    gasProt_dists = cdist(locs, positions)
    toDelete = np.where(np.min(gasProt_dists, axis=1) < 1.0)
    locs = np.delete(locs, toDelete, 0)

    # Find number of each gas molecule according to atmospheric composition where 0=N2 and 1=O2 (exclude trace gases)
    composition = np.random.choice([0, 1], size=len(locs), p=[0.786, 0.214])

    # Seed molecules
    for count, molec in enumerate(composition):
        # Nitrogen gas
        if molec == 0:
            res_num = (atoms[-1].res_num) + 1
            res_name = 'NNN'
            res_id = str(res_num) + res_name
            atom_names = ['NNN1', 'NNN2']
            init_coords = np.array([[0.0, 0.0, 0.0], [0.10977, 0.0, 0.0]])
            element = 'N'

        # Oxygen gas
        elif molec == 1:
            res_num = (atoms[-1].res_num) + 1
            res_name = 'OOO'
            res_id = str(res_num) + res_name
            atom_names = ['OOO1', 'OOO2']
            init_coords = np.array([[0.0, 0.0, 0.0], [0.12074, 0.0, 0.0]])
            element = 'O'            

        # Seed molecule
        atom_nums = [((atoms[-1].atom_num) + x) for x in range(2)]
        coords = init_coords + locs[count]
        for atom in range(2):
            new_atom = Atom(res_id, res_num, res_name, atom_names[atom], atom_nums[atom], coords[atom], None, element)
            atoms.append(new_atom)

    # Update coord file
    atoms = update_atoms(atoms, top_order)
    write_gro('droplet.gro', atoms, box_vectors)
    print('Atmosphere seeded.')


def form_droplet(atoms, residues, positions, box_vectors, args, top_order, proteins):
    """
    Droplet formation as specified by user defined args including, setting protonation states
    via Hend. Hass. for inputted protein .pdb, forming droplet of given size, seeding solutes
    such as ammonium acetate considering Rayleigh limit, and seeding atmosphere. 

    Args:
        atoms (list): Ordered list of Atom objects.
        residues (list): Ordered list of Residue objects.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        box_vectors (list): 3D coordinates defining box size.
        args: argparse object of user defined args.  
        top_order (list): Ordered list of residue names for .top files.
        proteins (list): Ordered list of Protein objects.

    Outputs:
        Complete droplet coordinate file (droplet.gro) and molecular topology (prot.top). 
        Residues (list): Corrected list of Residue objects.
        atoms (list): Corrected list of Atom objects.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
    """ 

    # Estimates initial droplet radius as 1.7x prot radius (assuming 1 g/cm^3 density)
    def dropletR_est(atoms):
        mass = sum(atom_masses[atom.element] for atom in atoms)
        prot_radius = ((3.0*((mass/Na)*(10**21)))/(4*math.pi*prot_density))**(1.0/3.0)
        return prot_radius*1.8

    # Finds # of ions corresponding to %90 the Rayleigh limit
    def find_numIons(droplet_radius):
        rayleigh_lim = (1.5866233*(10**20))*((5.216*(10**-13)*((droplet_radius*(10**-9))**3))**0.5)
        prot_charge = find_protCharge(proteins)
        if args.esi_mode == 'pos':
            numIons = round((rayleigh_lim*0.95)-prot_charge)
        elif args.esi_mode == 'neg':
            numIons = round((rayleigh_lim*0.95)+prot_charge)        
        return numIons

    # Carves out droplet from solvation box
    def carve_droplet(atoms, droplet_radius, num_waters):
        waters_removed = 0
        atoms_toDelete = []
        for index, atom in enumerate(atoms):
            if atom.res_name == 'SOL' and atom.atom_name == 'OW':
                if np.linalg.norm(np.subtract(box_center, atom.coord)) > droplet_radius:
                    waters_removed += 1
                    for i in range(4):
                        atoms_toDelete.append(index+i)
        atoms_toDelete.sort(reverse=True)
        for index in atoms_toDelete:
            del atoms[index]
        num_waters -= waters_removed
        return atoms, num_waters

    # Initial droplet size
    droplet_radius = args.droplet_size if args.droplet_size is not None else dropletR_est(atoms)

    # Begin droplet formation by finding how many ions to seed; H3O+ in pos mode, OH- in neg mode, or ammonium acetate.
    # Choose to seed droplet with user defined ions if --ion seeding set to manual, else 90% Rayleigh limit 
    num_molecs = {}
    box_center = np.array([vec/2 for vec in box_vectors])
    
    # Need to find number of droplet waters while taking into account displacement from the protein meaning we have to form a pure water droplet. 
    shutil.copyfile('prot.top', 'temp.top')
    subprocess.run(f'gmx solvate -cp prot.gro -cs tip4p_2005.gro -p temp.top -o solvated.gro', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, temp_residues, temp_atoms, _, _, _, _  = unpack_gro('solvated.gro')
    temp_num_waters = 0
    for residue in temp_residues:
        if residue.res_name == 'SOL':
            temp_num_waters += 1
    _, temp_num_waters = carve_droplet(temp_atoms, droplet_radius, temp_num_waters)

    # Find num of acetate molecules to seed dependent given molar concentration from --amace_conc, seed excess NH4+ in pos mode, acetate in neg mode
    num_amace = round(0.01801*temp_num_waters*args.amace_conc)
    num_molecs['HHO'] = 0
    num_molecs['OHX'] = 0
    if args.esi_mode == 'pos':
        num_molecs['ATX'] = round(num_amace*0.9) # At ph 5.5, CH3COOH: CH3COO- ratio is ~1:10
        num_molecs['AHX'] = round(num_amace*0.1)
        num_molecs['NXH'] = find_numIons(droplet_radius) + round(num_amace*0.9) # Add in excess NH4+ to satisfy Rayleigh limit
        num_molecs['NXX'] = 0
    elif args.esi_mode == 'neg':
        num_molecs['NXH'] = round(num_amace*0.9)
        num_molecs['NXX'] = round(num_amace*0.1)
        num_molecs['ATX'] = find_numIons(droplet_radius) + round(num_amace*0.9) # Add in excess acetate to satisfy Rayleigh limit
        num_molecs['AHX'] = 0

    # Set gas molecules to 0 for now
    num_molecs['NNN'] = 0 #N2
    num_molecs['OOO'] = 0 #O2

    ################# MOLECULE INSERTION #################

    # Seeds a given molecule into the system given that molecules .gro coord file
    def insert_molec(gro_file, insert_num, spacing, positions):
        # Get atom info from .gro file, .gro must be centered at 0, 0, 0!
        _, _, molec_atoms, _, _, _, _ = unpack_gro(gro_file)
        molec_coords = [atom.coord for atom in molec_atoms]

        # Find new molecule position
        attempts = 0
        for _ in range(insert_num):
            #iterates until finds position not overlapping w/ other molecules and near droplet surface given initial structure
            while True:
                #smaller radius so molecules don't 'jut' out from droplet and are fully solvated
                random_loc = np.random.uniform(box_center-(droplet_radius*0.9), box_center+(droplet_radius*0.9), 3)
                coords = np.add(molec_coords, random_loc)
                min_dist = min([np.min(np.linalg.norm(np.subtract(positions, coord), axis=-1)) for coord in coords])
                if min_dist > spacing and droplet_radius > np.linalg.norm(np.subtract(box_center, random_loc)):
                    positions = np.concatenate((positions, coords), axis=0)
                    break
                else:
                    attempts += 1
                    if attempts > 10000:
                        print('Molecule insertion failed after 10,000 attempts. Likely need larger droplet size to solvate protein.')
                        exit(1)

            #create new molecule
            res_num = (atoms[-1].res_num) + 1
            res_name = molec_atoms[0].res_name
            res_id = str(res_num) + res_name
            atom_nums = [((atoms[-1].atom_num) + x) for x in range(len(coords))]
            velocity = None
            new_residue = Residue(res_id, res_num, res_name)
            for index, atom in enumerate(molec_atoms):
                new_atom = Atom(res_id, res_num, atom.res_name, atom.atom_name, atom_nums[index], coords[index], velocity, atom.element)
                atoms.append(new_atom)
                new_residue.add_atom(new_atom)
            residues.append(new_residue)
        print(f'{res_name} insertion completed after {attempts} attempts.')
        print(f'Added {insert_num} {res_name} molecules.')

    # Choose to seed droplet with assocatiated molecs 
    if num_molecs['HHO'] > 0: #H3O+
        insert_molec('h3o.gro', num_molecs['HHO'], 1.0, positions)

    if num_molecs['OHX'] > 0: #OH-
        insert_molec('oh.gro', num_molecs['OHX'], 1.0, positions)

    if num_molecs['ATX'] > 0: #acetate
        insert_molec('ace.gro', num_molecs['ATX'], 0.35, positions)

    if num_molecs['AHX'] > 0: #acetic acid
        insert_molec('aceh.gro', num_molecs['AHX'], 0.35, positions)   

    if num_molecs['NXX'] > 0: #ammonia
        insert_molec('nh3.gro', num_molecs['NXX'], 0.35, positions)

    if num_molecs['NXH'] > 0: #ammonium
        insert_molec('nh4.gro', num_molecs['NXH'], 0.35, positions)

    # Set num waters to zero as we seed those using gromacs later
    num_molecs['SOL'] = 0

    # Modify top file for new molecules
    with open(f'prot.top', 'r') as f:
        data = f.readlines()
    for res_name in top_order:
        data.append(f'{res_name}                 {num_molecs[res_name]}\n')
    with open(f'prot.top', 'w') as f:
        data = "".join(data)
        f.write(data)

    # Write new coordinate file
    atoms = update_atoms(atoms, top_order)
    write_gro('prot.gro', atoms, box_vectors)

    # Solvate system
    subprocess.run(f'gmx solvate -cp prot.gro -cs tip4p_2005.gro -p prot.top -o solvated.gro', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, residues, atoms, box_vectors, _, _, _  = unpack_gro('solvated.gro')
    num_waters = sum(1 for residue in residues if residue.res_name == 'SOL')

    # Fixes .top, gmx solvate puts SOL line on bottom which we don't want so remove
    with open(f'prot.top', 'r') as f:
        data = f.readlines()
    data = data[:-1]
    with open(f'prot.top', 'w') as f:
        data = "".join(data)
        f.write(data)
    print('Protein solvated.')

    # Carve out droplet
    atoms, num_waters = carve_droplet(atoms, droplet_radius, num_waters)
    num_molecs['SOL'] = num_waters
    print(f'Droplet now contains {num_waters} waters.')  

    # Very specific ordering in top file or else grompp will reject
    modify_top('prot.top', top_order, num_molecs)
    atoms = update_atoms(atoms, top_order)

    # Write new .gro file
    write_gro('droplet.gro', atoms, box_vectors)

    # Expand box
    subprocess.run('gmx editconf -f droplet.gro -o droplet.gro -box 50 -c', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Add in atmospere if --atm enabled
    if args.atm:
        _, _, atoms, box_vectors, positions, _, _  = unpack_gro('droplet.gro')
        form_atmosphere(atoms, positions, box_vectors, top_order)

    # Update residues/atoms 
    _, residues, atoms, _, _, res_dict, num_molecs = unpack_gro('droplet.gro')

    # Create final droplet .top file with all molecules
    _, num_molecs = correct_dicts(res_dict, num_molecs, top_order)
    modify_top('prot.top', top_order, num_molecs)

    print('Droplet formation completed.')
    return residues, atoms, num_molecs