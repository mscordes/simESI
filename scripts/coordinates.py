'''
Series of functions for parsing and modifying molecular coordinate and topology files.
'''
import numpy as np
from string import digits, ascii_uppercase
import random
import os
from scipy.spatial.distance import cdist
from classes import Atom, Residue, Protein, Titratable_Sites
from gmx import auto_gmx_input

def unpack_gro(gro_file):
    """Extracts all information from a .gro file. Unpacks gro file into Atoms, Residue, and Proteins objects.
    Also stores box vectors of simulation box, atomic positions, and number of each molecule.

    Args:
        gro_file (str): Name of .gro file to unpack.

    Outputs:
        proteins (list): Ordered list of Protein objects.
        residues (list): Ordered list of Residue objects.
        atoms (list): Ordered list of Atom objects.
        box_vectors (list): 3D coordinates defining box size.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
    
    """
    res_dict = {}
    positions = []
    proteins = []
    residues = []
    atoms = []
    box_vectors = None
    remove_digits = str.maketrans('', '', digits)
    with open(gro_file) as f:
        gro = f.readlines()
        box_vectors = [float(vector) for vector in gro[-1].split()]
        residue = None
        current_res = None
        current_protein = Protein()
        res_count = -1
        prot_switch = False
        for index, line in enumerate(gro[2:-1]):
            res_temp = line[0:9].replace(" ", "")
            res_num = res_count
            res_name = res_temp.translate(remove_digits)[0:3]
            res_id = str(res_num)+res_name

            if current_res != res_temp:
                if residue is not None:
                    residues.append(residue)
                    current_protein.add_residue(residue)
                    if prot_switch:
                        proteins.append(current_protein)
                        current_protein = Protein()
                        prot_switch = False
                    try:
                        res_dict[residue.res_name].append(residue)
                    except KeyError:
                        res_dict[residue.res_name] = [residue]
                res_count += 1
                res_num = res_count
                res_id = str(res_num)+res_name
                residue = Residue(res_id, res_num, res_name)
                current_res = res_temp

            atom_name = line[9:15].replace(" ", "")
            atom_num = index
            coord = np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])])
            positions.append(coord)
            if not line[44:52].isspace():
                try:
                    velocity = np.array([float(line[44:52]), float(line[52:60]), float(line[60:68])])
                    max_vel = max(max(velocity), -min(velocity))
                    if max_vel > 3:
                        if res_name in ['NNN', 'OOO']:
                            velocity = atoms[index-2].velocity
                            for atom in residue.atoms: 
                                atom.velocity = atoms[index-2].velocity
                        elif max_vel > 10.0:
                            velocity = atoms[index-2].velocity
                except ValueError:
                    velocity = atoms[index-1].velocity
            else:
                velocity = None
            element = atom_name[0]
            atom = Atom(res_id, res_num, res_name, atom_name, atom_num, coord, velocity, element)
            atoms.append(atom)
            residue.add_atom(atom)
            current_protein.add_atom(atom)

            if atom.atom_name in ['OXT', 'OT2', 'HT2']:
                if atom.atom_name == 'OT2':
                    try:
                        if gro[2:-1][index+1].find('HT2') != -1:
                            pass
                        else:
                            prot_switch = True          
                    except IndexError:
                        prot_switch = True      
                else:
                    prot_switch = True

        try:
            res_dict[residue.res_name].append(residue)
        except KeyError:
            res_dict[residue.res_name] = [residue]
        residues.append(residue)
        
        if prot_switch == True:
            current_protein.add_residue(residue)
            proteins.append(current_protein)

        num_molecs = {key: len(res_dict[key]) for key in res_dict}
    return proteins, residues, atoms, box_vectors, np.array(positions), res_dict, num_molecs


def write_gro(filename, atoms, box_vectors):
    """Using stored atoms list and box vectors, write new, properly formatted .gro file.
    Also allows for the creation of .gro files with >100,000 atoms.
    
    Args:
        filename (str): Name of .gro file to create.
        atoms (list): Ordered list of Atom objects.
        box_vectors (list): 3D coordinates defining box size.

    Outputs:
        New .gro file as defined by {filename}.
    """
    with open(f'{filename}', 'w') as f:
        f.write('simESI Generated Coordinate File \n')
        f.write(f' {len(atoms)}\n')
        if atoms[0].velocity is not None:
            current_res = None
            res_count = 0
            atom_count = 1
            for atom in atoms:
                atom_count += 1
                if atom_count == 100000:
                    atom_count = 1
                if atom.res_id != current_res:
                    current_res = atom.res_id
                    res_count += 1
                    if res_count == 100000:
                        res_count = 1
                f.write(
                    '{:>5}'.format(res_count) + 
                    '{:<5}'.format(atom.res_name) +
                    '{:>5}'.format(atom.atom_name) +
                    '{:>5}'.format(atom_count) + 
                    '{:>8}'.format('{:.3f}'.format(atom.coord[0])) +
                    '{:>8}'.format('{:.3f}'.format(atom.coord[1])) +
                    '{:>8}'.format('{:.3f}'.format(atom.coord[2])) +
                    '{:>8}'.format('{:.3f}'.format(atom.velocity[0])) +
                    '{:>8}'.format('{:.3f}'.format(atom.velocity[1])) +
                    '{:>8}'.format('{:.3f}'.format(atom.velocity[2])) +
                    '\n'
                )
        else:
            current_res = None
            res_count = 0
            atom_count = 1
            for atom in atoms:
                atom_count += 1
                if atom_count == 100000:
                    atom_count = 1
                if atom.res_id != current_res:
                    current_res = atom.res_id
                    res_count += 1
                    if res_count == 100000:
                        res_count = 1
                f.write(
                    '{:>5}'.format(res_count) + 
                    '{:<5}'.format(atom.res_name) +
                    '{:>5}'.format(atom.atom_name) +
                    '{:>5}'.format(atom_count) + 
                    '{:>8}'.format('{:.3f}'.format(atom.coord[0])) +
                    '{:>8}'.format('{:.3f}'.format(atom.coord[1])) +
                    '{:>8}'.format('{:.3f}'.format(atom.coord[2])) +
                    '\n'
                )
        f.write(
            ' ' + str(box_vectors[0]) + 
            ' ' + str(box_vectors[1]) + 
            ' ' + str(box_vectors[2]) + 
            '\n'
        )


def unpack_pdb(pdb_file):
    """Extracts all information from a .pdb file. Unpacks pdb file into Atoms, Residue, and Proteins objects.
    Also stores box vectors of simulation box, atomic positions, and number of each molecule.

    Args:
        pdb_file (str): Name of .pdb file to unpack.

    Outputs:
        proteins (list): Ordered list of Protein objects.
        residues (list): Ordered list of Residue objects.
        atoms (list): Ordered list of Atom objects.
        box_vectors (list): 3D coordinates defining box size.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
        temp_factors (list): Ordered list of temp factors.
    """
    res_dict = {}
    positions = []
    proteins = []
    residues = []
    atoms = []
    temp_factors = []
    box_vectors = None
    with open(pdb_file) as f:
        pdb = f.readlines()
        for line in pdb:
            if line[0:6] == 'CRYST1':
                box_vectors = [float(vector)/10.0 for index, vector in enumerate(line.split()) if index in [1,2,3]]
                break
        residue = None
        current_res = None
        current_protein = Protein()
        res_count = -1
        prot_switch = False
        for index, line in enumerate(pdb):
            if 'ATOM' == line[0:4]:
                res_num = res_count
                res_name = line[16:21].replace(" ", "")
                res_temp = line[22:27].replace(" ", "") + res_name
                res_id = str(res_num)+res_name

                if current_res != res_temp:
                    if residue is not None:
                        residues.append(residue)
                        current_protein.add_residue(residue)
                        if prot_switch:
                            proteins.append(current_protein)
                            current_protein = Protein()
                            prot_switch = False
                        try:
                            res_dict[residue.res_name].append(residue)
                        except KeyError:
                            res_dict[residue.res_name] = [residue]
                    res_count += 1
                    res_num = res_count
                    res_id = str(res_num)+res_name
                    residue = Residue(res_id, res_num, res_name)
                    current_res = res_temp

                atom_name = line[12:17]
                atom_num = index
                coord = np.array([float(line[30:38].replace(" ", ""))/10.0, \
                                  float(line[38:46].replace(" ", ""))/10.0, \
                                  float(line[46:54].replace(" ", ""))/10.0])
                positions.append(coord)
                velocity = None
                element = atom_name[0]
                atom = Atom(res_id, res_num, res_name, atom_name, atom_num, coord, velocity, element)
                atoms.append(atom)
                temp_factors.append(line[60:67])
                residue.add_atom(atom)
                current_protein.add_atom(atom)
            elif line[0:3] == 'TER':
                prot_switch = True

        try:
            res_dict[residue.res_name].append(residue)
        except KeyError:
            res_dict[residue.res_name] = [residue]
        residues.append(residue)
        
        if prot_switch == True:
            current_protein.add_residue(residue)
            proteins.append(current_protein)

        num_molecs = {key: len(res_dict[key]) for key in res_dict}
    return proteins, residues, atoms, box_vectors, np.array(positions), res_dict, num_molecs, temp_factors


def write_pdb(filename, proteins, box_vectors, temp_factors):
    """Using stored proteins list and box vectors, write new, properly formatted .pdb file.
    .gro files don't store chain information which is necessary to simulated protein complexes!
    This file considers complexation, additionally, fixes residue numbering from restarting at 0 
    every new chain, to counting up which is necessary given that pka values are stored according to 
    residue number (i.e., no double repeats).

    Args:
        filename (str): Name of .pdb file to create.
        proteins (list): Ordered list of Protein objects.
        box_vectors (list): 3D coordinates defining box size.
        temp_factors (list): Ordered list of temp factors.

    Outputs:
        New .pdb file as defined by {filename}.
    """
    with open(f'{filename}', 'w') as f:
        f.write(f'CRYST1  {"{:.2f}".format(box_vectors[0]*10)}  {"{:.2f}".format(box_vectors[1]*10)}  {"{:.2f}".format(box_vectors[2]*10)}  90.00  90.00  90.00 P 1           1\n')
        f.write('MODEL        1 \n')
        for prot_index, protein in enumerate(proteins):
            for index, atom in enumerate(protein.atoms):
                atom_string = 'ATOM' + '{:>7}'.format(index+1)
                if len(atom.atom_name) > 3:
                    atom_string += ' ' + '{:<1}'.format(atom.atom_name.ljust(5))
                else:
                    atom_string += '  ' + '{:<2}'.format(atom.atom_name.ljust(5)) 
                atom_string += atom.res_name.ljust(4) 
                atom_string += ascii_uppercase[prot_index]
                atom_string +='{:>4}'.format(atom.res_num + 1) 
                atom_string +='{:>12}'.format("{:.3f}".format(atom.coord[0]*10)) 
                atom_string +='{:>8}'.format("{:.3f}".format(atom.coord[1]*10)) 
                atom_string +='{:>8}'.format("{:.3f}".format(atom.coord[2]*10)) 
                atom_string +='{:>6}'.format('1.00') 
                atom_string +='{:>6}'.format(temp_factors[index]) 
                atom_string +='{:>11}'.format(atom.element) 
                atom_string += '\n'
                f.write(atom_string)
            f.write('TER\n')
        
        f.write('ENDMDL\n')


def set_bond_length(coord1, coord2, length):
    """Moves coordinate of coord2 to a desired distance away from coord1
    along the axis between the two points. This is used to transfer a proton
    from donor to acceptor during a proton transfer.

    Args:
        coord1 (np array): Reference 3D coord of atom1.
        coord2 (np array): 3D coord of atom2 to be moved.
        length (float): Desired distance between coord1 and coord2.

    Outputs:
        new_coord (np array): New 3D coord of coord2 at specified length from coord1.
    """
    disp_vec = coord2 - coord1
    axis_dir = disp_vec / np.linalg.norm(disp_vec)
    new_coord = coord1 + length * axis_dir
    return new_coord


def update_atoms(atoms, top_order):
    """gmx requires consistent ordering of atoms in .top file. After facilitating exchanges, order
    can get screwed up so correct here.

    Args:
        atoms (list): Ordered list of Atom objects.
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        atoms (list): Corrected, ordered list of Atom objects.
    """
    # Dict with key = resiude name (as defined in .top), value = all atoms of that residue
    res_dict = {res: [] for res in top_order}
    res_dict['Protein'] = []
    res_dict_keys = res_dict.keys()

    # Sort atoms to key in res_dict
    for atom in atoms:
        if atom.res_name in res_dict_keys:
            res_dict[atom.res_name].append(atom)
        else:
            res_dict['Protein'].append(atom)

    # Re-order atoms
    atoms = res_dict['Protein']
    for res in top_order:
        atoms = atoms + res_dict[res]
    return atoms


def correct_dicts(res_dict, num_molecs, top_order):
    """If residue not present in system, populates res_dict/num_molecs with key value pairs with val of 0.

    Args:
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        res_dict (dict): Corrected, dict where each key is a residue type, and value is ordered list 
        of residues of that type.
        num_molecs (dict): Corrected, dict with keys as residue names, and values corresponding to the 
        number of that residue present in the simulation.
    """
    for res in top_order:
        if res not in num_molecs:
            res_dict[res] = []
            num_molecs[res] = 0
    return res_dict, num_molecs


def get_protAtoms(atoms, top_order):
    """Gets atoms and coords of atoms specific to protein.

    Args:
        atoms (list): Ordered list of Atom objects.
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        protein_atoms (list): Ordered list of Atom objects corresponding to protein atoms.
        Ordered np array of 3D coordinates of all protein atoms.
    """
    protein_atoms = []
    protein_coords = []
    for atom in atoms:
        if atom.res_name not in top_order:
            protein_atoms.append(atom)
            protein_coords.append(atom.coord)
        else:
            break
    return protein_atoms, np.array(protein_coords)


def get_waterCoords(waters):
    """Gets atoms and coords of all water hydrogens and oxygens.

    Args:
        waters (list): Ordered list of water Residue objects.

    Outputs:
        Ordered np array of 3D coordinates of all water oxygens.
        Ordered np array of 3D coordinates of all water hydrogens.
        waterO_atoms (list): Ordered list of water oxygen Atom objects.
        waterH_atoms (list): Ordered list of water hydrogen Atom objects.
    """
    if len(waters) > 0:
        #creates np array of water coords split into O/H
        waterO_coords = []
        waterO_atoms = []
        waterH_coords = []
        waterH_atoms = []
        for residue in waters:
            res_atoms = residue.atoms
            waterO_atoms.append(res_atoms[0])
            waterO_coords.append(res_atoms[0].coord)
            waterH_atoms.append(res_atoms[1])
            waterH_atoms.append(res_atoms[2])
            waterH_coords.append(res_atoms[1].coord)
            waterH_coords.append(res_atoms[2].coord)
        return np.array(waterO_coords), np.array(waterH_coords), waterO_atoms, waterH_atoms
    else:
        return np.array([]), np.array([]), [], []


def fix_disulfides(residues, atoms):
    """Annoying bug where will pdb2gmx will not recognize disulfide bond if bond extends beyond 2.0 ± 0.2 Å.
    Not only do we want to keep disulfides intact, but simESI requires every atom be accounted for so this 
    bug actually breaks the simulation as well.

    Args:
        residues (list): Ordered list of Residue objects.
        atoms (list): Ordered list of Atom objects.

    Outputs:
        atoms (list): Corrected, ordered list of Atom objects.
    """
    # Find CYS sulfurs
    ss_atom_indices = []
    for res in residues:
        if res.res_name == 'CYS':
            for atom in res.atoms:
                if atom.atom_name == 'SG':
                    ss_atom_indices.append(atom.atom_num)
        elif res.res_name in ['SOL', 'NNN']:
            break
    ss_atoms = [atoms[index] for index in ss_atom_indices]

    # Find sulfurs that can participate in a SS-bridge
    ss_bridges = []
    for atom1 in ss_atoms:
        for atom2 in ss_atoms:
            if atom1 != atom2 and atom2 not in [pair[0] for pair in ss_bridges] and np.linalg.norm(atom1.coord - atom2.coord) < 0.25:
                ss_bridges.append([atom1, atom2])

    # If SS bond length short/long, reset length to 2.05 Å along existing bond axis
    for bridge in ss_bridges:
        dist = np.linalg.norm(bridge[0].coord - bridge[1].coord)
        if dist > 0.21 or dist < 0.19:
            # Use center of SS bond as ref
            com = np.mean(np.array([bridge[0].coord, bridge[1].coord]), axis=0)
            # Update atoms list with new coord
            atoms[bridge[0].atom_num].coord = set_bond_length(com, bridge[0].coord, 0.1025)
            atoms[bridge[1].atom_num].coord = set_bond_length(com, bridge[1].coord, 0.1025)
    return atoms


def set_protonMap(res_dict, proteins, pka_vals):
    """Returns a 'proton_map', an ordered list of amino acid protonation states as an input to gmx pdb2gmx
    with values computed via Henderson Hasselbach and amino acid pKa's (from PROPKA) assuming an initial pH of 7.

    Args:
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        proteins (list): Ordered list of Protein objects.
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
             residue number, and values corresponding to its pKa as computed via PROPKA.

    Outputs:
        proton_map (list): Ordered list of gmx pdb2gmx inputs for a specific pattern of protonation
        (taking into account protein complexation!).
    """

    # Finds protonation state of residue, returns value encoding that state for the proton_map given an inputted pka
    def set_protState(protein, proton_map, res_dict, resname, prot_val, deprot_val):
        res_proton_map = []
        try:
            res_list = res_dict[resname]
        except KeyError:
            return proton_map
        
        # Bounds tells us if residue within the protein monomer in question (if protein a complex)
        lower_bound = protein.residues[0].res_num
        upper_bound = protein.residues[-1].res_num
        for residue in res_list:
            if residue.res_num >= lower_bound and residue.res_num <= upper_bound:
                # Probility resiude protonated at ph 7
                try:
                    prob = 1.0/(1+(10**(7.0-pka_vals[residue.res_num])))
                    prot_state = random.choices([1, 0], weights=(prob, 1-prob), k=1)[0]
                except ValueError:
                    print("\n\n\nERROR: PROPKA .pka file was generated, but is likely missing a residue."
                        "A common issue is that the C-termini for each monomer must have both oxygens, and the "
                        "final oxygen must have the atom name 'OXT'not 'OT1' or 'OT2'.")
                    exit(1)

                # Update proton map
                if prot_state == 1:
                    res_proton_map.append(prot_val)
                else:
                    res_proton_map.append(deprot_val)
            else:
                pass
        
        #Update master proton map
        proton_map += res_proton_map
        return proton_map

    total_proton_map = []
    for protein in proteins:
        proton_map = []
        #Find sites of each titratable amino acid (including termini)
        proton_map = set_protState(protein, proton_map, res_dict, 'LYS', 1, 0)
        proton_map = set_protState(protein, proton_map, res_dict, 'ARG', 1, 0)
        proton_map = set_protState(protein, proton_map, res_dict, 'ASP', 1, 0)
        proton_map = set_protState(protein, proton_map, res_dict, 'GLU', 1, 0)
        proton_map = set_protState(protein, proton_map, res_dict, 'HIS', 2, 0)

        # Termini
        res_dict['nterm'] = [protein.residues[0]]
        res_dict['cterm'] = [protein.residues[-1]]
        #inputs for termini, GROMACS defines different protonation assignments for nterm for MET/PRO
        if protein.residues[0].res_name == 'MET':
            nterm_prot_val = 1 
            nterm_deprot_val = 2
        elif protein.residues[0].res_name == 'PRO':
            nterm_prot_val = 1 
            nterm_deprot_val = 0
        else:
            nterm_prot_val = 0 
            nterm_deprot_val = 1
        proton_map = set_protState(protein, proton_map, res_dict, 'nterm', nterm_prot_val, nterm_deprot_val)
        proton_map = set_protState(protein, proton_map, res_dict, 'cterm', 1, 0)
        total_proton_map.append(proton_map)
    return total_proton_map


def get_charges(residues, proteins):
    """Gets partial charges of all atoms.

    Args:
        residues (list): Ordered list of Residue objects.
        proteins (list): Ordered list of Protein objects.

    Outputs:
        np array of partial charges.
    """
    charges = []
    for protein in range(len(proteins)):
        with open(f'monomer_{protein}.itp', 'r') as f:
            top_data = [line.split() for line in f.readlines()]
            for line in top_data:
                if len(line) > 7:
                    try:
                        charges.append(float(line[6]))
                    except ValueError:
                        pass
    for residue in residues:
        if residue.res_name == 'SOL':
            charges.extend((0.00000, 0.55640, 0.55640, -1.11280))
        elif residue.res_name == 'HHO':
            charges.extend((0.00000, 0.41600, 0.41600, -0.24800, 0.41600))
        elif residue.res_name == 'OHX':
            charges.extend((-1.32000, 0.32000))
        elif residue.res_name == 'NNN':
            charges.extend((0.00000, 0.00000))
        elif residue.res_name == 'OOO':
            charges.extend((0.00000, 0.00000))
        elif residue.res_name == 'ATX':
            charges.extend((-0.37000, 0.62000, 0.09000, 0.09000, 0.09000, -0.76000, -0.76000))
        elif residue.res_name == 'AHX':
            charges.extend((-0.30000, 0.75000, 0.09000, 0.09000, 0.09000, -0.55000, -0.60000, 0.43000))
        elif residue.res_name == 'NXX':
            charges.extend((-1.12500, 0.37500, 0.37500, 0.37500))
        elif residue.res_name == 'NXH':
            charges.extend((0.33000, -0.32000, 0.33000, 0.33000, 0.33000))
    return np.array(charges)


def find_protCharge(proteins):
    """Finds net protein charge (considering complexation).

    Args:
        proteins (list): Ordered list of Protein objects.

    Outputs:
        charge (int): Net protein (or protein complex) charge.
    """
    charge = 0
    for monomer in range(len(proteins)):
        with open(f'monomer_{monomer}.itp', 'r') as f:
            data = f.readlines()

        cutoff = None
        for index, line in enumerate(data):
            if '[ bonds ]' in line:
                cutoff = index
                break

        for line in range(cutoff, 0, -1):
            if 'qtot' in data[line]:
                charge += int(data[line].split()[-1])
                break
    return charge


def convert_top2itp(prot_index):
    """gmx pdb2gmx creates a .top file for a protein. simESI generates these for every protein
    mononmer which we can then convert to individual .itp files that can be rolled into a single, 
    master .top file to allow for protein complexes.

    Args:
        prot_index (int): Number associated to monomer. I.e., first protein is 0, second is 1, etc.

    Outputs:
        Outputs corrected .itp file of protein monomer.
    """
    indices_toDelete = []
    with open(f'monomer_{prot_index}.top', 'r') as f:
        top = f.readlines()
        for indx, line in enumerate(reversed(top)):
            if line.find('; Include Position restraint file') == -1:
                indices_toDelete.append(len(top)-indx-1)
            else:
                indices_toDelete.append(len(top)-indx-1)
                break
        for indx, line in enumerate(top):
            if line.find('[ moleculetype ]') == -1:
                indices_toDelete.append(indx)
            else:
                break
    indices_toDelete.sort(reverse=True)
    for indx in indices_toDelete:
        del top[indx]
    top[2] = f'monomer_{prot_index}          3'
    with open(f'monomer_{prot_index}.itp', 'w') as f:
        top = "".join(top)
        f.write(top)


def create_top(fname, proteins, top_order, num_molecs):
    """Create new .top file from scratch with necessary forcefield info, number of each molecules, 
    and .itp's corresponding to each protein monomer.

    Args:
        fname (str): Filename for newly created .top file.
        proteins (list): Ordered list of Protein objects.
        top_order (list): Ordered list of residue names for .top files.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.

    Outputs:
        Outputs new, valid .top file.
    """
    data = []
    data.append('#include "charmm36.ff/forcefield.itp"\n')
    for monomer in range(len(proteins)):
        data.append(f'#include "monomer_{monomer}.itp"\n')
    data.append('#include "o2.itp"\n')     
    data.append('#include "n2.itp"\n')
    data.append('#include "nh4.itp"\n')
    data.append('#include "nh3.itp"\n') 
    data.append('#include "aceh.itp"\n') 
    data.append('#include "ace.itp"\n')
    data.append('#include "hydroxide.itp"\n')
    data.append('#include "hydronium.itp"\n')
    data.append('#include "tip4p_2005.itp"\n')
    data.append('[ system ]\n')
    data.append('Electrosprayed Droplet\n\n')
    data.append('[ molecules ]\n')
    for monomer in range(len(proteins)):
        data.append(f'monomer_{monomer}    1\n')
    if top_order is not None:
        for res_name in top_order:
            try:
                data.append(f'{res_name}                 {num_molecs[res_name]}\n')
            except KeyError:
                data.append(f'{res_name}                 {0}\n')
    with open(f'{fname}', 'w') as f:
        data = "".join(data)
        f.write(data)
        

def modify_top(top_file, top_order, num_molecs):
    """Updates number of each residue in .top file.

    Args:
        top_name (str): Filename for modified .top file.
        top_order (list): Ordered list of residue names for .top files.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.

    Outputs:
        Outputs modified .top file with name as inputted via {top_file} arg.
    """
    with open(f'{top_file}', 'r') as f:
        data = f.readlines()
    for index, res in enumerate(reversed(top_order)):
        data[-index-1] = f'{res}                 {num_molecs[res]}\n'
    with open(f'{top_file}', 'w') as f:
        data = "".join(data)
        f.write(data)


def write_top(proteins, outp_top, top_order, num_molecs, proton_map, args, box_vectors, create_gro):
    """Creates new .top preserving protein protonation states from inputted coordinates.

    Args:
        proteins (list): Ordered list of Protein objects.
        outp_top (str): Filename for newly created .top file.
        top_order (list): Ordered list of residue names for .top files.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
        proton_map (list): Ordered list of gmx pdb2gmx inputs for a specific pattern of protonation.
        args: argparse object of user defined args.
        box_vectors (list): 3D coordinates defining box size.
        create_gro (bool): If True, Write new .gro file. This only necessary on the initial call.

    Outputs:
        Outputs new .top file with name defined by {outp_top}.
        Optionally, 
    """
    for index, monomer in enumerate(proteins):
        write_gro(f'monomer_{index}.gro', monomer.atoms, box_vectors)
        #sets protonations states
        if os.path.exists(outp_top):
            os.remove(outp_top)
        if create_gro:
            command = f'gmx pdb2gmx -f monomer_{index}.gro -o monomer_{index}.gro -p monomer_{index}.top -ff charmm36 -water none -lys -arg -glu -asp -his -ter -ignh'
            rem_fname = f'#monomer_{index}.gro.1#'
        else:
            command = f'gmx pdb2gmx -f monomer_{index}.gro -o temp1.gro -p monomer_{index}.top -ff charmm36 -water none -lys -arg -glu -asp -his -ter -ignh'
            rem_fname = 'temp1.gro'
        auto_gmx_input(command, proton_map[index], args)
        os.remove('posre.itp')
        os.remove(rem_fname)

        #convert .top to .itp
        convert_top2itp(index)
        os.remove(f'monomer_{index}.top')

    # Write new .top
    create_top(outp_top, proteins, top_order, num_molecs)

    # Optionally write new .gro 
    if create_gro:
        atoms_master = []
        for index in range(len(proteins)):
            _, _, mon_atoms, box_vectors, _, _, _ = unpack_gro(f'monomer_{index}.gro')
            atoms_master += mon_atoms
        write_gro('prot.gro', atoms_master, box_vectors)


def get_protonMap(res_dict, proteins):
    """Returns a 'proton_map', an ordered list of amino acid protonation states as an input to gmx pdb2gmx 
    considering the protonation states of the inputted protein class Objects.

    Args:
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        proteins (list): Ordered list of Protein objects.

    Outputs:
        total_proton_map (list): Ordered list of gmx pdb2gmx inputs for a specific pattern of protonation 
        (considering protein complexes!).
    """

    # Finds protonation state of residue, returns value encoding that state for the proton_map
    def find_protState(protein, proton_map, res_dict, resname, tit_hydrogen, tit_acceptor, max_hydrogens, prot_val, deprot_val):
        res_proton_map = []
        try:
            res_list = res_dict[resname]
        except KeyError:
            return proton_map
        
        # Bounds tells us if residue within the protein monomer in question (if protein a complex)
        lower_bound = protein.residues[0].res_num
        upper_bound = protein.residues[-1].res_num
        for residue in res_list:
            if residue.res_num >= lower_bound and residue.res_num <= upper_bound:
                donor = []
                acceptor = []
                for atom in residue.atoms:
                    if atom.atom_name in tit_hydrogen:
                        donor.append(atom)
                    elif atom.atom_name in tit_acceptor:
                        acceptor.append(atom)
                if donor and len(donor) == max_hydrogens:
                    res_proton_map.append(prot_val)
                else:
                    #HIS special in that only one of thier 2 protonable sites can accept
                    if atom.res_name.find('HIS') != -1:
                        if 'HD1' in [x.atom_name for x in donor]:
                            res_proton_map.append(0)
                            acceptor = [x for x in acceptor if x.atom_name != 'ND1']
                        elif 'HE2' in [x.atom_name for x in donor]:
                            res_proton_map.append(1)
                            acceptor = [x for x in acceptor if x.atom_name != 'NE2']
                    else:
                        res_proton_map.append(deprot_val)
            else:
                pass
        
        #Update master proton map
        proton_map += res_proton_map
        return proton_map

    total_proton_map = []
    for protein in proteins:
        proton_map = []
        #Find sites of each titratable amino acid (including termini)
        proton_map = find_protState(protein, proton_map, res_dict, 'LYS', ['HZ1', 'HZ2', 'HZ3'], ['NZ'], 3, 1, 0)
        proton_map = find_protState(protein, proton_map, res_dict, 'ARG', ['HH11', 'HH12', 'HH21', 'HH22'], ['NH1'], 4, 1, 0)
        proton_map = find_protState(protein, proton_map, res_dict, 'ASP', ['HD2'], ['OD1', 'OD2'], 1, 1, 0)
        proton_map = find_protState(protein, proton_map, res_dict, 'GLU', ['HE2'], ['OE1', 'OE2'], 1, 1, 0)
        proton_map = find_protState(protein, proton_map, res_dict, 'HIS', ['HD1', 'HE2'], ['NE2', 'ND1'], 2, 2, 0)

        # Termini
        res_dict['nterm'] = [protein.residues[0]]
        res_dict['cterm'] = [protein.residues[-1]]
        #inputs for termini, GROMACS defines different protonation assignments for nterm for MET/PRO
        if protein.residues[0].res_name == 'MET':
            nterm_prot_val = 1 
            nterm_deprot_val = 2
        elif protein.residues[0].res_name == 'PRO':
            nterm_prot_val = 1 
            nterm_deprot_val = 0
        else:
            nterm_prot_val = 0 
            nterm_deprot_val = 1
        proton_map = find_protState(protein, proton_map, res_dict, 'nterm', ['H1', 'H2', 'H3'], ['N'], 3, nterm_prot_val, nterm_deprot_val)
        proton_map = find_protState(protein, proton_map, res_dict, 'cterm', ['HT2'], ['OT1', 'OT2'], 1, 1, 0)
        total_proton_map.append(proton_map)
    return total_proton_map


def pin_cluster(res_coords, waterO_coords, clusters, cluster_waters):
    """Determines which cluster is solvating each residue of a given resiude type.

    Args:
        res_coords (np array): Ordered array of all coords of a particular residue type.    
        waterO_coords (np array): Ordered array of all water oxygen coords.
        clusters (np array): Ordered list where each val is the corresponding cluster label of an individual
            water residue, in the same order as the waters in the coordinate file.
        cluster_waters (dict): Number of waters in each cluster given cluster label as key, num water as val.

    Outputs:
        near waters (list): Ordered list of number of waters in pinned cluster to each residue.
        pinned_clusters (lsit): Ordered list of cluster label of the pinned cluster to each residue.
    """
    if res_coords.size != 0 and len(waterO_coords) > 0:
        distances = cdist(res_coords, waterO_coords)
        closest_water_indices = np.argmin(distances, axis=1) 
        closest_water_distances = np.min(distances, axis=1)
        pinned_clusters = []
        near_waters = []
        for index, min_dist in enumerate(closest_water_distances):
            if min_dist < 0.50:
                ind_cluster = clusters[closest_water_indices[index]]
                pinned_clusters.append(ind_cluster)

                if ind_cluster == -1: #outliers
                    near_waters.append(np.count_nonzero(distances[index] < 1.0))
                else:
                    near_waters.append(cluster_waters[ind_cluster])
            else:
                pinned_clusters.append(-2) #-2 is code for gas phase residues
                near_waters.append(0)
    else:
        near_waters = [0 for _ in res_coords]
        pinned_clusters = [-2 for _ in res_coords]
    return near_waters, pinned_clusters


def get_titSites(tit_sites, system, clusters, cluster_waters, prot_flag):
    """Parses and stores titratable sites of all protonable resiudes by packing them into Titrable_Sites objects. 
    Returns dict where keys are residue names (i.e. LYS for all lysine residues) and vals are Titratable_Sites objects.

    Args:
        tit_sites (dict): Dict where keys are residue names, vals are Titratable_Sites objects. Can be empty
            dict if creating or previously defined tit_sites dict.
        system: System class object.    
        clusters (lsit): Ordered list of cluster label of the pinned cluster to each residue.
        cluster_waters (list): Ordered list of number of waters in pinned cluster to each residue.
        prot_flag (bool): If True, parse protein titrable sites. This is expensive, so only done when
            necessary.

    Outputs:
        tit_sites (dict): Corrected tit_sites dict.
        system.res_dict (dict): Updates System class objects res_dict where dict keys are residue types, and 
        vals are ordered list of residues of that type.
    """

    # Finds tit sites of indiviudal amino acid type and return Titrable_Sites class / proton map
    def find_titSite(res_dict, resname, tit_hydrogen, tit_acceptor, max_hydrogens):
        prot_atoms = []
        deprot_atoms = []
        try:
            res_list = res_dict[resname]
        except KeyError:
            return Titratable_Sites([], [], [], [], [], [], [], [])     

        for residue in res_list:
            donor = []
            acceptor = []
            for atom in residue.atoms:
                if atom.atom_name in tit_hydrogen:
                    donor.append(atom)
                elif atom.atom_name in tit_acceptor:
                    acceptor.append(atom)
            if donor and len(donor) == max_hydrogens:
                prot_atoms += donor
            else:
                #HIS special in that only one of thier 2 protonable sites can accept
                if atom.res_name.find('HIS') != -1:
                    if 'HD1' in [x.atom_name for x in donor]:
                        acceptor = [x for x in acceptor if x.atom_name != 'ND1']
                    elif 'HE2' in [x.atom_name for x in donor]:
                        acceptor = [x for x in acceptor if x.atom_name != 'NE2']
                    deprot_atoms += acceptor
                else:
                    deprot_atoms += acceptor

        #H-bonding and convert coords to array
        prot_coords = np.array([atom.coord for atom in prot_atoms])
        deprot_coords = np.array([atom.coord for atom in deprot_atoms])
        prot_nearWaters, prot_clusters = pin_cluster(prot_coords, system.waterO_coords, clusters, cluster_waters)
        deprot_nearWaters, deprot_clusters = pin_cluster(deprot_coords, system.waterO_coords, clusters, cluster_waters)
        return Titratable_Sites(prot_atoms, prot_coords, prot_nearWaters, prot_clusters, \
                                deprot_atoms, deprot_coords, deprot_nearWaters, deprot_clusters)

    # Finds all non-protein titrable sites
    def find_nonProt_titSite(res_dict, resname, tit_index, protonated):
        try:
            res_list = res_dict[resname]
        except KeyError:
            return Titratable_Sites([], [], [], [], [], [], [], [])
        
        tit_atoms = []
        for residue in res_list:
            for index in tit_index:
                tit_atoms.append(residue.atoms[index])
        tit_coords = np.array([atom.coord for atom in tit_atoms])

        near_waters, res_clusters = pin_cluster(tit_coords, system.waterO_coords, clusters, cluster_waters)

        if protonated:
            return Titratable_Sites(tit_atoms, tit_coords, near_waters, res_clusters, [], [], [], [])
        else:
            return Titratable_Sites([], [], [], [], tit_atoms, tit_coords, near_waters, res_clusters)

    # Finding tit sites of prot expensive, so only do when necessary
    if prot_flag:
        #Find sites of each titratable amino acid (including termini)
        tit_sites['LYS'] = find_titSite(system.res_dict, 'LYS', ['HZ1', 'HZ2', 'HZ3'], ['NZ'], 3)
        tit_sites['ARG'] = find_titSite(system.res_dict, 'ARG', ['HH11', 'HH12', 'HH21', 'HH22'], ['NH1'], 4)
        tit_sites['ASP'] = find_titSite(system.res_dict, 'ASP', ['HD2'], ['OD1', 'OD2'], 1)
        tit_sites['GLU'] = find_titSite(system.res_dict, 'GLU', ['HE2'], ['OE1', 'OE2'], 1)
        tit_sites['HIS'] = find_titSite(system.res_dict, 'HIS', ['HD1', 'HE2'], ['NE2', 'ND1'], 2)

        # Termini
        system.res_dict['nterm'] = []
        system.res_dict['cterm'] = []
        for protein in system.proteins:  
            system.res_dict['nterm'].append(protein.residues[0])
            system.res_dict['cterm'].append(protein.residues[-1])

        tit_sites['nterm'] = find_titSite(system.res_dict, 'nterm', ['H1', 'H2', 'H3'], ['N'], 3)
        tit_sites['cterm'] = find_titSite(system.res_dict, 'cterm', ['HT2'], ['OT1', 'OT2'], 1)

    #Final titratable sites
    tit_sites['HHO'] = find_nonProt_titSite(system.res_dict, 'HHO', [1, 2, 4], True) #H3O+
    tit_sites['OHX'] = find_nonProt_titSite(system.res_dict, 'OHX', [0], False) #OH-
    tit_sites['ATX'] = find_nonProt_titSite(system.res_dict, 'ATX', [5,6], False) #acetate
    tit_sites['AHX'] = find_nonProt_titSite(system.res_dict, 'AHX', [7], True) #acetic acid
    tit_sites['NXX'] = find_nonProt_titSite(system.res_dict, 'NXX', [0], False) #NH3
    tit_sites['NXH'] = find_nonProt_titSite(system.res_dict, 'NXH', [0, 2, 3, 4], True) #NH4+

    #Add in water
    waterO_clusters = clusters
    waterO_nearWaters = [cluster_waters[cluster] for cluster in clusters]
    waterH_clusters = [x for x in waterO_clusters for _ in range(2)]
    waterH_nearWaters = [x for x in waterO_nearWaters for _ in range(2)]
    tit_sites['SOL'] = Titratable_Sites(system.waterH_atoms, system.waterH_coords, waterH_nearWaters, waterH_clusters, \
                                        system.waterO_atoms, system.waterO_coords, waterO_nearWaters, waterO_clusters)
    return tit_sites, system.res_dict


def top_from_coord(coordfile, top_name, args, top_order):
    """This generates a valid .top file from given coords while preserving protonation states.
    Very useful for continuation of a run.

    Args:
        coordfile (str): Filename of coordinate file to parse. 
        top_name (str): Filename of .top file to generate.
        args: argparse object of user defined args.  
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        Newly generated, valid .top file with correct protononation states.
    """

    # Unpack .gro
    proteins, _, _, box_vectors, _, res_dict, num_molecs = unpack_gro(f'{coordfile}')
    res_dict, num_molecs = correct_dicts(res_dict, num_molecs, top_order)

    # Get map of protonation states as input for pdb2gmx when .top is written
    proton_map = get_protonMap(res_dict, proteins)

    # Write new .gro and .top file
    write_top(proteins, f'{top_name}', top_order, num_molecs, proton_map, args, box_vectors, False)