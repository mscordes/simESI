import numpy as np

def remove_evaporated(system, args):
    """Deletes evaporated molecules.

    Args:
        system: System class object.
        args: Argparse object of user defined args.

    Outputs:
        system: Corrected System class object with evaporated molecs removed.
        com (np array): 3D coord of center of mass of system. 
        evap (bool): Flag of whether or not any molecules were deleted (as this 
            requires changing the molecular topology).
    """

    # Find center-of-mass of droplet (ignore atmosphere if present)
    if args.atm:
        cutoff = system.res_dict['NNN'][0].atoms[0].atom_num
    else:
        cutoff = len(system.atoms)
    atom_masses = {'O':15.99, 'C': 12.000, 'N':14.007, 'H':1.0078, 'S':32.065, 'M':0.0}
    masses = np.array([atom_masses[atom.element] for atom in system.atoms[:cutoff]])
    nonGas_coords = system.positions[:cutoff]
    com = np.sum(masses[:, np.newaxis]*nonGas_coords, axis=0) / np.sum(masses)

    # Indices to delete
    atoms_toDelete = []

    # Delete evaporated waters
    # WARNING: ALL evaporated waters must be deleted prior to clustering as outliers severly mess with HDBSCAN clustering
    if system.num_molecs['SOL'] > 0:
        water_dist = np.linalg.norm(np.subtract(system.waterO_coords, com), axis=-1)

        if len(system.res_dict['SOL']) > 100: # Outlier detection to remove waters in system with more than 100 waters
            d = np.abs(water_dist - np.median(water_dist))
            mdev = np.median(d)
            s = d/mdev if mdev else np.zeros(len(d))
            mask = s > 6.0

        else: # Outlier detection begins to fail with smaller systems so simple 5.0 nm cutoff here
            mask = water_dist > 5.0

        # Log evaporated waters for deletion and update number of waters
        evap_indices = np.where(mask)[0]
        for index in evap_indices:
            for atom in system.res_dict['SOL'][index].atoms:
                atoms_toDelete.append(atom.atom_num)
        system.num_molecs['SOL'] = len(system.res_dict['SOL']) - len(evap_indices)

    def find_evap(resname, com, cutoff, res_dict, num_molecs):
        if res_dict[resname]:
            evaporated = 0
            for res in res_dict[resname]:
                if np.linalg.norm(res.atoms[0].coord - com) > cutoff:
                    evaporated += 1
                    for atom in res.atoms:
                        atoms_toDelete.append(atom.atom_num) 
            system.num_molecs[resname] = len(res_dict[resname]) - evaporated
        else:
            system.num_molecs[resname] = 0
        return atoms_toDelete, num_molecs

    # Delete evaporated ions
    cutoff = 12.5 # This works for smallish proteins, but will need to be increased for larger
    for res in ['HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
        atoms_toDelete, system.num_molecs = find_evap(res, com, cutoff, system.res_dict, system.num_molecs)

    # Update atoms list
    evap = False
    if atoms_toDelete:
        atoms_toDelete.sort(reverse=True)
        for indx in atoms_toDelete:
            del system.atoms[indx]
        evap = True
    return system, com, evap 