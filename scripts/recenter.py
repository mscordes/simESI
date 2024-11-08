import numpy as np
from scipy.stats import maxwell, kstest

def recenter_droplet(atoms, residues, box_vectors, poly_com, res_dict):
    """Recenters droplet if it drifts towards edge of box. Very important if atm enabled as droplet 
    tends to drift significantly and droplet cannot interact with PBC.

    Args:
        atoms (list): Ordered list of Atom objects.
        residues (list): Ordered list of Residue objects.
        box_vectors (list): 3D coordinates defining box size.
        poly_com (np array): 3D coord of center of mass of protein.
        res_dict (dict): Dict where each key is a residue type, and value is ordered list of residues
            of that type.

    Outputs:
        atoms (list): Corrected, ordered list of Atom objects.
    """
    com = np.array(box_vectors)/2
    translation = poly_com - com
    for atom in atoms:
        atom.coord -= translation
        mask_large = atom.coord > box_vectors[0]
        mask_small = atom.coord < 0.0
        if np.any(mask_large) or np.any(mask_small):
            res_atoms = residues[atom.res_num].atoms
            for res_atom in res_atoms:
                atoms[res_atom.atom_num].coord[mask_large] -= box_vectors[0]
                atoms[res_atom.atom_num].coord[mask_small] += box_vectors[0]
            
    # Need to correct for gas molecules that are still outside box
    nitrogens = [atom for res in res_dict['NNN'] for atom in res.atoms]
    oxygens = [atom for res in res_dict['OOO'] for atom in res.atoms]
    gas_atoms = nitrogens + oxygens
    moved = []
    for atom in gas_atoms:
        mask_large = atom.coord > box_vectors[0]
        mask_small = atom.coord < 0.0
        if np.any(mask_large) == True or np.any(mask_small) == True:
            res_atoms = residues[atom.res_num].atoms
            for res_atom in res_atoms:
                atoms[res_atom.atom_num].coord[mask_large] -= 0.2
                atoms[res_atom.atom_num].coord[mask_small] += 0.2
                moved.append(atoms[res_atom.atom_num])
    
    # Because of gas molecules moved above, now have to correct for steric clashes with other gas molecs
    gas_coords = np.array([atoms[atom.atom_num].coord for atom in gas_atoms])
    iteration = 0
    good_atoms = []
    while True:
        clashes = 0
        for atom in moved:
            if atom.atom_num not in good_atoms:
                dist = np.linalg.norm(np.subtract(gas_coords, atoms[atom.atom_num].coord), axis=-1)
                idx = np.argpartition(dist, 3)
                mins = dist[idx[:3]] < 1.0
                if mins.sum() == 3:
                    # Move slightly towards center of box (to avoid pushing outside) until clash removed
                    vector = poly_com - atom.coord
                    translation = vector*(0.1/np.linalg.norm(vector))
                    res_atoms = residues[atom.res_num].atoms
                    for res_atom in res_atoms:                    
                        atoms[res_atom.atom_num].coord = atoms[res_atom.atom_num].coord + translation
                        gas_coords = np.array([atoms[atom.atom_num].coord for atom in gas_atoms])
                        clashes += 1
                else:
                    good_atoms.append(atom.atom_num)
        iteration += 1
        if clashes == 0 or iteration == 100:
            break
    return atoms


def reset_gasVelocities(system, gas_correct):
    """Between stitches, rarely, gas molecule velocities can become distorted
    with a couple of gas molecules consuming all energy of the T coupling
    (i,e., moving incredibly fast) while all other molecules are essentially
    frozen. Correct here by determining how close the actual velocity distr.
    is compared to expected. If distorted, gas velocities are reseeded to conform
    to an expected distribution. This is equivalent to gen_vel = yes in .mdp params,
    that is specifically applied to only atmospheric molecules.

    Args:
        system: System class object.
        gas_correct (bool): Flag if gas velocities were corrected, if so, need 
            to generate new .tpr file.

    Outputs:
        system.atoms (list): Corrected, ordered list of Atom objects with fixed velocities.
        gas_correct (bool)
    """
    for gas in ['NNN', 'OOO']:
        # Actual gas speed distribution
        gas_v = np.array([atom.velocity for res in system.res_dict[gas] for atom in res.atoms])
        gas_speed = np.linalg.norm(gas_v, axis=1)*1000 # Convert to velocity in m/s
        gas_dist, bin_edges = np.histogram(gas_speed, bins=20, density=True)

        # Predicted gas vel dist. from Maxwell-Boltzmann
        temperature = 300 #Gas temp (K)
        mass = 14.0067 if gas == 'NNN' else 15.999
        scale = np.sqrt((1000*8.314*temperature)/mass)
        midpoints = (bin_edges[:-1] + bin_edges[1:])/2
        b_dist = maxwell.pdf(midpoints, scale=scale)

        # Kolmogorov–Smirnov test to determine if gas vel dist. good
        if kstest(gas_dist, b_dist).pvalue < 0.95:
            gas_correct = True

            # Correct gas velocities sampled from Maxwell-Boltzmann
            newGas_speed = maxwell.rvs(scale=scale, size=len(gas_speed))
            newGas_speed /= 1000 #convert from m/s to nm/ps
            newGas_speed = np.reshape(newGas_speed, (-1, 1))

            # Generate a random direction uniformly on the unit sphere
            theta = np.random.uniform(0, 2*np.pi, len(newGas_speed)) #Azimuthal angle
            phi = np.random.uniform(0, np.pi, len(newGas_speed)) #Polar angle
            
            # Convert spherical coordinates into unit vectors
            x = np.sin(phi)*np.cos(theta)
            y = np.sin(phi)*np.sin(theta)
            z = np.cos(phi)
            unit_vectors = np.array([x, y, z]).T
            
            # Scale by the desired speed
            newGas_v = unit_vectors*newGas_speed

            # Update atom velocities
            count = 0
            for res in system.res_dict[gas]:
                for atom in res.atoms:
                    atom.velocity = newGas_v[count]
                    count += 1
    return system.atoms, gas_correct