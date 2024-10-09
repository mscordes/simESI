'''
Gets system info via various functions from coordinates.py and packs all system information into a System class object.
Also defines functions to efficiently update the System object.
'''
import numpy as np
from coordinates import *
from classes import Residue, Protein, System

def sys_info(gro_filename, top_order):
    """Finds all info for, and outputs complete system object via numerous
    functions defined in coordinate.py.

    Args:
        gro_filename (str): .gro filename to be parsed.
        top_order (list): Ordered list of residue names for .top files.

    Outputs:
        Complete System class object.
    """
    proteins, residues, atoms, box_vectors, positions, res_dict, num_molecs = unpack_gro(f'{gro_filename}')
    res_dict, num_molecs = correct_dicts(res_dict, num_molecs, top_order)
    protein_atoms, protein_coords = get_protAtoms(atoms, top_order)
    charges = get_charges(residues, proteins)
    prot_charge = find_protCharge(proteins)
    waterO_coords, waterH_coords, waterO_atoms, waterH_atoms = get_waterCoords(res_dict['SOL'])
    sys_charge = prot_charge + num_molecs['HHO'] - num_molecs['OHX'] + num_molecs['NXH'] - num_molecs['ATX']
    return System(proteins, residues, atoms, box_vectors, positions, res_dict, num_molecs, protein_atoms, protein_coords, \
                    charges, prot_charge, waterO_coords, waterH_coords, waterO_atoms, waterH_atoms, sys_charge)


def update_pos_resDict_numMolecs(atoms):
    """This updates various System info with post-exchanged coordinates. This is an abbreviated version of unpack_gro() 
    from coordinates.py with string manipulation removed as atoms already in object class, so much quicker.

    Args:
        atoms (list): Ordered list of atom class objects.

    Outputs:
        proteins (list): Ordered list of Protein objects.
        residues (list): Ordered list of Residue objects.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
    """
    res_dict = {}
    positions = []
    residues = []
    residue = None
    proteins = []
    current_protein = Protein()
    current_res = None
    prot_switch = False
    res_count = -1
    for index, atom in enumerate(atoms):
        positions.append(atom.coord)
        atom.atom_num = index
        if current_res != atom.res_id:
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
            res_name = atom.res_name
            res_id = str(res_count)+res_name
            residue = Residue(res_id, res_count, res_name)
            current_res = atom.res_id
        atom.atom_num = index
        atom.res_id = res_id
        atom.res_num = res_count
        residue.add_atom(atom)
        current_protein.add_atom(atom)
        if atom.atom_name in ['OXT', 'OT2', 'HT2']:
            if atom.atom_name == 'OT2':
                try:
                    if atoms[index+1].atom_name == 'HT2':
                        pass
                    else:
                        prot_switch = True          
                except IndexError:
                    prot_switch = True      
            else:
                prot_switch = True
    residues.append(residue)
    try:
        res_dict[residue.res_name].append(residue)
    except KeyError:
        res_dict[residue.res_name] = [residue]
    if prot_switch == True:
        current_protein.add_residue(residue)
        proteins.append(current_protein)
    num_molecs = {key: len(res_dict[key]) for key in res_dict}
    return proteins, residues, np.array(positions), res_dict, num_molecs


def update_system(system, args, top_order, step, prot):
    """Update System class object to reflect proton exchanges dependent on type that were facilitated.

    Args:
        system: System class object.
        args: argparse object of user defined args. 
        top_order (list): Ordered list of residue names for .top files.
        prot_flag (bool): Flag, True if exchanges with protein were facilitated.
        
    Outputs:
        system: Corrected System class object.
    """
    # Update residues, positions, res_dict, num_molecs
    system.proteins, system.residues, system.positions, system.res_dict, system.num_molecs = update_pos_resDict_numMolecs(system.atoms)
    system.res_dict, system.num_molecs = correct_dicts(system.res_dict, system.num_molecs, top_order)

    # If exchanges with protein, must create new .top file, this is a somewhat expensive procedure
    if prot:
        system.protein_atoms, system.protein_coords = get_protAtoms(system.atoms, top_order)
        proton_map = get_protonMap(system.res_dict, system.proteins)
        write_top(system.proteins, f'{step}.top', top_order, system.num_molecs, proton_map, args, system.box_vectors, False)
        system.prot_charge = find_protCharge(system.proteins)
        system.charges = get_charges(system.residues, system.proteins)
        system.waterO_coords, system.waterH_coords, system.waterO_atoms, system.waterH_atoms = get_waterCoords(system.res_dict['SOL'])
    else:
        modify_top(f'{step}.top', top_order, system.num_molecs)
        
    # Finish updating sysytem with charges/water coords
    system.charges = get_charges(system.residues, system.proteins)
    system.waterO_coords, system.waterH_coords, system.waterO_atoms, system.waterH_atoms = get_waterCoords(system.res_dict['SOL'])
    return system 