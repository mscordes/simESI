'''
Facilitates proton exchanges and outputs corrected, post-exchange atoms.
 * Proton exchanges are facilitated by topology switching the product and reactant for Grotthuss exchanges
 * For all other protonations, make new molecule with transferred proton and delete old reactant.
 * For all other deprotonations, simply delete exchanged H and topology switch to deprotonated state. 
 * Updates coordinates (including transforms for water/H3O+) to ensure proper post-exchange structures
'''
import numpy as np
from classes import Atom
from copy import deepcopy
from scipy.optimize import minimize
from coordinates import update_atoms, set_bond_length
from transform import hydronium_transform, water_transform, angle
    
def grotthuss_H3O(atoms, pair):
    """Updates coordinates for H3O+ + water -> water + H3O+ Grotthuss exchanges. 
    Grotthuss exchanges are simplistic in that reactants/products same, so can simply
    switch coordinates of reactants/products.

    Args:
        atoms (list): Ordered list of Atom class objects.
        pair (Exchange class object): Proton donor-acceptor pair

    Outputs:
       atoms (list): Post exchange, updated list of Atom class objects. 
    """
    hydrogen = pair.donor
    water_O = pair.acceptor
    water_OIndex = water_O.atom_num

    # Switches position of hydrogen that will be exchanged from position 1 -> 3 as only 3rd position switched
    hydronium_OIndex = None
    if hydrogen.atom_name == 'HW1':
        temp = atoms[hydrogen.atom_num].coord
        atoms[hydrogen.atom_num].coord = atoms[hydrogen.atom_num+3].coord
        atoms[hydrogen.atom_num+3].coord = temp
        hydronium_OIndex = hydrogen.atom_num-1
    elif hydrogen.atom_name == 'HW2':
        temp = atoms[hydrogen.atom_num].coord
        atoms[hydrogen.atom_num].coord = atoms[hydrogen.atom_num+2].coord
        atoms[hydrogen.atom_num+2].coord = temp
        hydronium_OIndex = hydrogen.atom_num-2
    elif hydrogen.atom_name == 'HW3':
        hydronium_OIndex = hydrogen.atom_num-4

    # Actual proton transfer by switching coordinates of first 4 hydronium atoms with 4 atoms of water selected for transfer
    for atom in range(4):
        temp = atoms[hydronium_OIndex+atom].coord
        atoms[hydronium_OIndex+atom].coord = atoms[water_OIndex+atom].coord
        atoms[water_OIndex+atom].coord = temp

    # Updates newly formed H3O & H2O proper geometry
    new_hydronium = [atoms[hydronium_OIndex+x] for x in range(5)]
    atoms = hydronium_transform(new_hydronium, atoms)
    new_water = [atoms[water_OIndex+x] for x in range(4)]
    atoms = water_transform(new_water, atoms)
    return atoms


def grotthuss_OH(atoms, pair):
    """Updates coordinates for OH- + water -> water + OH- Grotthuss exchanges.
    Grotthuss exchanges are simplistic in that reactants/products same, so can simply
    switch coordinates of reactants/products.

    Args:
        atoms (list): Ordered list of Atom class objects.
        pair (Exchange class object): Proton donor-acceptor pair

    Outputs:
       atoms (list): Post exchange, updated list of Atom class objects. 
    """
    hydrogen = pair.donor
    oxygen = pair.acceptor
    water_O = None
    non_exchangedH = None
    if hydrogen.atom_name == 'HW2':
        water_O = atoms[hydrogen.atom_num-1]
        non_exchangedH = atoms[hydrogen.atom_num+1]
    elif hydrogen.atom_name == 'HW3':
        water_O = atoms[hydrogen.atom_num-2]
        non_exchangedH = atoms[hydrogen.atom_num-1]

    # Actual proton transfer by switching 2 non-exchanged coords startin with oxygen
    temp = atoms[oxygen.atom_num].coord
    atoms[oxygen.atom_num].coord = water_O.coord
    atoms[water_O.atom_num].coord = temp

    # Switch for H's
    temp = atoms[oxygen.atom_num+1].coord
    atoms[oxygen.atom_num+1].coord = non_exchangedH.coord
    atoms[non_exchangedH.atom_num].coord = temp

    # Updates newly formed H2O proper geometry
    new_water = [atoms[water_O.atom_num+x] for x in range(4)]
    atoms = water_transform(new_water, atoms)
    return atoms


def findMPM_H_loc(hydrogen, acceptor, positions):
    """Annoying bug where if carboxylate protonated at low angle, will break simulation. This optimization function
    finds a new, reasonable hydrogen location.

    Args:
        hydrogen (Atom object): Hydrogen being transferred.
        acceptor (Atom object): Proton acceptor being transferred.
        positions (np array): Ordered array of 3D coordinates of all atoms.

    Outputs:
        3D coords of new hydrogen location. 
    """

    # Find nearby atoms excluding accepting site
    new_positions = deepcopy(positions)
    new_positions = np.delete(new_positions, acceptor.atom_num, 0)
    distances = np.linalg.norm(acceptor.coord - new_positions, axis=-1)
    mask = distances < 0.30
    nearby_positions = new_positions[mask]

    # Optimization func to determine optimal loc based on electrostatic potential
    def cost_function(new_point):
        dist = np.linalg.norm(new_point - acceptor.coord)
        bond_penalty = (abs(dist - 0.10))**2
        lj_dist = 0.18/np.linalg.norm(nearby_positions - new_point, axis=-1)
        lj_potential = np.sum(lj_dist**12 - lj_dist**6)
        return 1.0*bond_penalty + 0.01*lj_potential
    
    #Initial guess for H loc
    initial_guess = set_bond_length(acceptor.coord, hydrogen.coord, 0.10)

    # Optimize
    optimization = minimize(cost_function, initial_guess, method='BFGS')
    return optimization.x


def create_protH(residue, hydrogen, atoms, hydrogens_toAdd, hydrogen_coord, nterm_resIDs, cterm_resIDs, new_HNames):
    """Updates coordinates for an amino acid protonation.

    Args:
        residue (Atom class object): Amino acid atom accepting the proton. 
        hydrogen (Atom class object): Hydrogen donor to be transferred to amino acid.
        atoms (list): Ordered list of Atom class objects.
        hydrogens_toAdd (list): List of all hydrogen Atom class objects being transferred to protein. Messing
            with protein protonation states requires special considerations done later so store them for now.
        hydrogen_coord (np 3D coord or None): Sometimes new hydrogen location precomputed. If so use that
            use that coord, else None to compute loc here.
        nterm_resIDs (list): List of residue ID's (residue number + type, i.e., 0LYS). Termini can sometimes have 2
            protonation sites (think Lys). If thats the case, have to determine if exchange is for terminal site, or 
            titratable site.
        cterm_resIDs (list): Same as nterm_resIDs, except for C-termini.
        new_HNames (list): List of residue names of newly formed hydrogens. These are specific to each amino acid type!

    Outputs:
       atoms (list): Post exchange, updated list of Atom objects.
       hydrogens_toAdd (list): Post exchange, updated list. 
    """
    if residue.res_id in nterm_resIDs and residue.atom_name == 'N':
        atom_name = new_HNames[residue.res_id]            
    elif residue.res_id in cterm_resIDs and residue.atom_name in ['OT1', 'OT2']:
        atom_name = new_HNames[residue.res_id]                
    elif residue.res_name.find('HIS') == -1:
        atom_name = new_HNames[residue.res_name]
    else:
        if residue.atom_name == 'ND1':
            atom_name = 'HD1'
            #CHARMM changes the atom order of HISH relative to HISE/HISD for no apparent fucking reason so have to correct
            atoms[residue.atom_num],   atoms[residue.atom_num+6] = atoms[residue.atom_num+6], atoms[residue.atom_num]
            atoms[residue.atom_num+1], atoms[residue.atom_num+7] = atoms[residue.atom_num+7], atoms[residue.atom_num+1]
            atoms[residue.atom_num+2], atoms[residue.atom_num+7] = atoms[residue.atom_num+7], atoms[residue.atom_num+2]
            atoms[residue.atom_num+3], atoms[residue.atom_num+4] = atoms[residue.atom_num+4], atoms[residue.atom_num+3]
            atoms[residue.atom_num+4], atoms[residue.atom_num+5] = atoms[residue.atom_num+5], atoms[residue.atom_num+4]
            atoms[residue.atom_num+5], atoms[residue.atom_num+6] = atoms[residue.atom_num+6], atoms[residue.atom_num+5]
            atoms[residue.atom_num+6], atoms[residue.atom_num+7] = atoms[residue.atom_num+7], atoms[residue.atom_num+6] 
        elif residue.atom_name == 'NE2':
            atom_name = 'HE2'
            atoms[residue.atom_num-5], atoms[residue.atom_num+1] = atoms[residue.atom_num+1], atoms[residue.atom_num-5]
            atoms[residue.atom_num-4], atoms[residue.atom_num+2] = atoms[residue.atom_num+2], atoms[residue.atom_num-4]
            atoms[residue.atom_num-2], atoms[residue.atom_num]   = atoms[residue.atom_num],   atoms[residue.atom_num-2]
            atoms[residue.atom_num-1], atoms[residue.atom_num+1] = atoms[residue.atom_num+1], atoms[residue.atom_num-1]
            atoms[residue.atom_num],   atoms[residue.atom_num+2] = atoms[residue.atom_num+2], atoms[residue.atom_num]
            atoms[residue.atom_num+1], atoms[residue.atom_num+2] = atoms[residue.atom_num+2], atoms[residue.atom_num+1]   

    # Create new H
    if hydrogen_coord is None:
        hydrogens_toAdd.append(Atom(residue.res_id, residue.res_num, residue.res_name, atom_name, 99999, \
                                set_bond_length(residue.coord, hydrogen.coord, 0.1), hydrogen.velocity, 'H'))
    else:
        hydrogens_toAdd.append(Atom(residue.res_id, residue.res_num, residue.res_name, atom_name, 99999, \
                                hydrogen_coord, hydrogen.velocity, 'H'))
    
    # Either carboxylate O in the C-termini
    if residue.res_id in cterm_resIDs and residue.atom_name in ['OT1', 'OT2']:
        if residue.atom_name == 'OT1':
            temp = residue.coord
            residue.coord = atoms[residue.atom_num+1].coord
            atoms[residue.atom_num+1].coord = temp
    # This enables exchange at any site in residues with multiple protonation sites starting with carboxylate O in GLU
    elif residue.res_name.find('GLU') != -1:
        if residue.atom_name == 'OE1':
            temp = residue.coord
            residue.coord = atoms[residue.atom_num+1].coord
            atoms[residue.atom_num+1].coord = temp
    # Now either carboxylate O in ASP 
    elif residue.res_name.find('ASP') != -1:
        if residue.atom_name == 'OD1':
            temp = residue.coord
            residue.coord = atoms[residue.atom_num+1].coord
            atoms[residue.atom_num+1].coord = temp
    return atoms, hydrogens_toAdd


def deprotonate_prot(hydrogen, atoms, atoms_toDelete):
    """Updates coordinates for an amino acid deprotonation.

    Args:
        hydrogen (Atom class object): Hydrogen donor to be transferred away from amino acid.
        atoms (list): Ordered list of Atom class objects.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.

    Outputs:
       atoms (list): Post exchange, updated list of Atom objects.
       atoms_toDelete (list): Post exchange, updated list.
    """
    atoms_toDelete.append(hydrogen)

    # Ntermini
    if hydrogen.atom_name in ['H1', 'H2', 'H3']:
        if hydrogen.atom_name == 'H1':
            atoms[hydrogen.atom_num+1].atom_name = 'H1'
            atoms[hydrogen.atom_num+2].atom_name = 'H2'
        elif hydrogen.atom_name == 'H2':
            atoms[hydrogen.atom_num+1].atom_name = 'H2'
    # Ctermini
    elif hydrogen.atom_name == 'HT2':
        pass          
    # This enables deprotonation of any protonable H in residues with multiple, starting with LYS
    elif hydrogen.res_name.find('LYS') != -1:
        if hydrogen.atom_name == 'HZ1':
            atoms[hydrogen.atom_num+1].atom_name = 'HZ1'
            atoms[hydrogen.atom_num+2].atom_name = 'HZ2'
        elif hydrogen.atom_name == 'HZ2':
            atoms[hydrogen.atom_num+1].atom_name = 'HZ2'
    # ARG is a pain, you have to switch a fair number of residues because only NH1 can be deprotonated
    elif hydrogen.res_name.find('ARG') != -1:
        if hydrogen.atom_name == 'HH11':
            atoms[hydrogen.atom_num+1].atom_name = 'HH11'
        elif hydrogen.atom_name == 'HH21':
            atoms[hydrogen.atom_num-4].atom_name = 'NH2'
            atoms[hydrogen.atom_num-1].atom_name = 'NH1'
            atoms[hydrogen.atom_num-4], atoms[hydrogen.atom_num-1] = atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num-4] 
            atoms[hydrogen.atom_num+1].atom_name = 'HH11'
            atoms[hydrogen.atom_num-3].atom_name = 'HH22'
            atoms[hydrogen.atom_num+1], atoms[hydrogen.atom_num-3] = atoms[hydrogen.atom_num-3], atoms[hydrogen.atom_num+1] 
            atoms[hydrogen.atom_num-2].atom_name = 'HH21'
            atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num-2] = atoms[hydrogen.atom_num-2], atoms[hydrogen.atom_num-1] 
        elif hydrogen.atom_name == 'HH22':
            atoms[hydrogen.atom_num-5].atom_name = 'NH2'
            atoms[hydrogen.atom_num-2].atom_name = 'NH1'
            atoms[hydrogen.atom_num-5], atoms[hydrogen.atom_num-2] = atoms[hydrogen.atom_num-2], atoms[hydrogen.atom_num-5] 
            atoms[hydrogen.atom_num-1].atom_name = 'HH11'
            atoms[hydrogen.atom_num-4].atom_name = 'HH22'
            atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num-4] = atoms[hydrogen.atom_num-4], atoms[hydrogen.atom_num-1] 
            atoms[hydrogen.atom_num-3].atom_name = 'HH21'
            atoms[hydrogen.atom_num-2], atoms[hydrogen.atom_num-3] = atoms[hydrogen.atom_num-3], atoms[hydrogen.atom_num-2] 
    # CHARMM changes the atom of order of HISH relative to HISE/HISD for no apparent fucking reason so have to correct
    elif hydrogen.res_name.find('HIS') != -1:
        if hydrogen.atom_name == 'HD1':
            atoms[hydrogen.atom_num-6], atoms[hydrogen.atom_num-1] = atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num-6]
            atoms[hydrogen.atom_num-5], atoms[hydrogen.atom_num-4] = atoms[hydrogen.atom_num-4], atoms[hydrogen.atom_num-5]
            atoms[hydrogen.atom_num-4], atoms[hydrogen.atom_num+1] = atoms[hydrogen.atom_num+1], atoms[hydrogen.atom_num-4]
            atoms[hydrogen.atom_num-3], atoms[hydrogen.atom_num+2] = atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num-3]
            atoms[hydrogen.atom_num-2], atoms[hydrogen.atom_num+2] = atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num-2]
            atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num+2] = atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num-1]
            atoms[hydrogen.atom_num+1], atoms[hydrogen.atom_num+2] = atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num+1]   
        elif hydrogen.atom_name == 'HE2':
            atoms[hydrogen.atom_num-4], atoms[hydrogen.atom_num+1] = atoms[hydrogen.atom_num+1], atoms[hydrogen.atom_num-4]
            atoms[hydrogen.atom_num-3], atoms[hydrogen.atom_num+2] = atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num-3]
            atoms[hydrogen.atom_num-1], atoms[hydrogen.atom_num+3] = atoms[hydrogen.atom_num+3], atoms[hydrogen.atom_num-1]
            atoms[hydrogen.atom_num+1], atoms[hydrogen.atom_num+4] = atoms[hydrogen.atom_num+4], atoms[hydrogen.atom_num+1]
            atoms[hydrogen.atom_num+2], atoms[hydrogen.atom_num+3] = atoms[hydrogen.atom_num+3], atoms[hydrogen.atom_num+2]
            atoms[hydrogen.atom_num+3], atoms[hydrogen.atom_num+4] = atoms[hydrogen.atom_num+4], atoms[hydrogen.atom_num+3]
    return atoms, atoms_toDelete


def create_water_from_hydronium(hydrogen, products_toAdd, atoms_toDelete, res_count, residues):
    """Deprotonate hydronium to create a newly formed water product.

    Args:
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old hydronium 
    for atom in residues[hydrogen.res_num].atoms:
        atoms_toDelete.append(atom)
    
    # Begin creating new water
    new_water = []
    res_count += 1
    # Start with oxygen
    old_oxygen = residues[hydrogen.res_num].atoms[0]
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'OW1', 99999, old_oxygen.coord, old_oxygen.velocity, 'O')) 
    # Water hydrogens
    if hydrogen.atom_name == 'HW1':
        nonExchanged_H1 = residues[hydrogen.res_num].atoms[2]
        nonExchanged_H2 = residues[hydrogen.res_num].atoms[4]
    elif hydrogen.atom_name == 'HW2':
        nonExchanged_H1 = residues[hydrogen.res_num].atoms[1]
        nonExchanged_H2 = residues[hydrogen.res_num].atoms[4]
    elif hydrogen.atom_name == 'HW3':
        nonExchanged_H1 = residues[hydrogen.res_num].atoms[1]
        nonExchanged_H2 = residues[hydrogen.res_num].atoms[2]
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'HW2', 99999, nonExchanged_H1.coord, nonExchanged_H1.velocity, 'H'))
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'HW3', 99999, nonExchanged_H2.coord, nonExchanged_H2.velocity, 'H'))
    # Water TIP4P virtual site
    old_vsite = residues[hydrogen.res_num].atoms[3]
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'MW4', 99999, old_vsite.coord, old_vsite.velocity, 'M')) 

    # Geometry transform to ensure proper water structure
    new_water = water_transform(new_water, None)
    for atom in new_water:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_hydronium_from_water(water, hydrogen, products_toAdd, atoms_toDelete, res_count, residues):
    """Protonate water to create a newly formed hydronium product.

    Args:
        water (Atom class object): Water oxygen, accepting site.
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old water
    for atom in residues[water.res_num].atoms:
        atoms_toDelete.append(atom)

    # Form new H3O from old water
    new_H3O = []
    res_count += 1
    for index, atom_name in enumerate(['OW', 'HW1', 'HW2', 'MW']):
        old_atom = residues[water.res_num].atoms[index]
        new_H3O.append(Atom(str(res_count)+'HHO', res_count, 'HHO', atom_name, 99999, old_atom.coord, old_atom.velocity, atom_name[0]))

    # Add former protein hydrogen to water to finish H3O formation
    new_H3O.append(Atom(str(res_count)+'HHO', res_count, 'HHO', 'HW3', 99999, set_bond_length(water.coord, hydrogen.coord, 0.09686), hydrogen.velocity, 'H'))

    # Geometry transform to ensure proper water structure
    new_H3O = hydronium_transform(new_H3O, None)
    for atom in new_H3O:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_water_from_hydroxide(hydroxide, hydrogen, products_toAdd, atoms_toDelete, res_count, residues):
    """Protonate hydroxide to create a newly formed water product.

    Args:
        hydroxide (Atom class object): Hydroxide oxygen, accepting site.
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old hydroxide
    for atom in residues[hydroxide.res_num].atoms:
        atoms_toDelete.append(atom)

    # Form new water from hydroxide
    new_water = []
    res_count += 1
    # Start with oxygen
    old_oxygen = residues[hydroxide.res_num].atoms[0]
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'OW1', 99999, old_oxygen.coord, old_oxygen.velocity, 'O')) 
    # Non-exchanged hydrogen
    nonExchanged_H = residues[hydroxide.res_num].atoms[1]
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'HW2', 99999, nonExchanged_H.coord, nonExchanged_H.velocity, 'H'))
    # Exchanged hydrogen
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'HW3', 99999, set_bond_length(hydroxide.coord, hydrogen.coord, 0.09686), old_oxygen.velocity, 'O'))
    # Water TIP4P virtual site, we'll set coord later in transform
    new_water.append(Atom(str(res_count)+'SOL', res_count, 'SOL', 'MW4', 99999, np.array([0.0, 0.0, 0.0]), old_oxygen.velocity, 'M'))      

    # Geometry transform to ensure proper water structure
    new_water = water_transform(new_water, None)
    for atom in new_water:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_hydroxide_from_water(hydrogen, products_toAdd, atoms_toDelete, res_count, residues):
    """Deprotonate water to create a newly formed hydroxude product.

    Args:
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old water
    for atom in residues[hydrogen.res_num].atoms:
        atoms_toDelete.append(atom)
    
    # Create new hydroxide, starting with oxygen
    res_count += 1
    old_oxygen = residues[hydrogen.res_num].atoms[0]
    products_toAdd.append(Atom(str(res_count)+'OHX', res_count, 'OHX', 'O1', 99999, old_oxygen.coord, old_oxygen.velocity, 'O'))

    # Hydroxide hydrogen
    if hydrogen.atom_name == 'HW2':
        nonExchanged_H = residues[hydrogen.res_num].atoms[2]
    elif hydrogen.atom_name == 'HW3':
        nonExchanged_H = residues[hydrogen.res_num].atoms[1]
    products_toAdd.append(Atom(str(res_count)+'OHX', res_count, 'OHX', 'H1', 99999, nonExchanged_H.coord, nonExchanged_H.velocity, 'H'))
    return products_toAdd, atoms_toDelete, res_count


def create_aceticAcid(acetate, hydrogen, products_toAdd, atoms_toDelete, res_count, residues, new_Hloc):
    """Protonate acetate to create a newly formed acetic acid product.

    Args:
        acetate (Atom class object): Acetate oxygen, accepting site.
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.
        new_Hloc (np array, optional): Precomputed new hydrogen loc if given, else, find loc here.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old acetate
    for atom in residues[acetate.res_num].atoms:
        atoms_toDelete.append(atom)

    # Create acetic acid
    res_count += 1
    new_aceticAcid = []
    res_atoms = residues[acetate.res_num].atoms
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'C2', 99999, res_atoms[0].coord, res_atoms[0].velocity, res_atoms[0].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'C1', 99999, res_atoms[1].coord, res_atoms[1].velocity, res_atoms[1].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'H21', 99999, res_atoms[2].coord, res_atoms[2].velocity, res_atoms[2].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'H22', 99999, res_atoms[3].coord, res_atoms[3].velocity, res_atoms[3].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'H23', 99999, res_atoms[4].coord, res_atoms[4].velocity, res_atoms[4].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'O2', 99999, res_atoms[5].coord, res_atoms[5].velocity, res_atoms[5].element))
    new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'O1', 99999, res_atoms[6].coord, res_atoms[6].velocity, res_atoms[6].element))

    # Allow for protonation of either carboxylate O
    if acetate.atom_name == 'O1':
        new_aceticAcid[5].atom_name = 'O1'
        new_aceticAcid[6].atom_name = 'O2'
        new_aceticAcid[5], new_aceticAcid[6] = new_aceticAcid[6], new_aceticAcid[5]

    # Add new hydrogen to acetate
    if new_Hloc is None:
        new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'HO1', 99999, set_bond_length(acetate.coord, hydrogen.coord, 0.1), hydrogen.velocity, hydrogen.element))
    else:
        new_aceticAcid.append(Atom(str(res_count)+'AHX', res_count, 'AHX', 'HO1', 99999, new_Hloc, hydrogen.velocity, hydrogen.element))

    # Finish product
    for atom in new_aceticAcid:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_ammonium(ammonia, hydrogen, products_toAdd, atoms_toDelete, res_count, residues):
    """Protonate ammonia to create a newly formed ammonium product.

    Args:
        ammonia (Atom class object): Ammonia nitrogen, accepting site.
        hydrogen (Atom class object): Hydrogen donor to be transferred.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old ammonia acid
    for atom in residues[ammonia.res_num].atoms:
        atoms_toDelete.append(atom)

    # Create ammonium
    res_count += 1
    new_ammonium = []
    res_atoms = residues[ammonia.res_num].atoms
    new_ammonium.append(Atom(str(res_count)+'NXH', res_count, 'NXH', 'NZ', 99999, res_atoms[0].coord, res_atoms[0].velocity, res_atoms[0].element))
    new_ammonium.append(Atom(str(res_count)+'NXH', res_count, 'NXH', 'HZ2', 99999, res_atoms[1].coord, res_atoms[1].velocity, res_atoms[1].element))
    new_ammonium.append(Atom(str(res_count)+'NXH', res_count, 'NXH', 'HZ3', 99999, res_atoms[2].coord, res_atoms[2].velocity, res_atoms[2].element))
    new_ammonium.append(Atom(str(res_count)+'NXH', res_count, 'NXH', 'HZ4', 99999, res_atoms[3].coord, res_atoms[3].velocity, res_atoms[3].element))

    # New hydrogen
    new_ammonium.insert(0, Atom(str(res_count)+'NXH', res_count, 'NXH', 'HZ1', 99999, set_bond_length(ammonia.coord, hydrogen.coord, 0.1), hydrogen.velocity, hydrogen.element))

    # Finish product
    for atom in new_ammonium:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_acetate(acetic_acid, products_toAdd, atoms_toDelete, res_count, residues):
    """Deprotonate acetic acid to create a newly formed acetate product.

    Args:
        acetate (Atom class object): Acetate oxygen, accepting site.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old acetic acid
    for atom in residues[acetic_acid.res_num].atoms:
        atoms_toDelete.append(atom)

    # Create acetate
    res_count += 1
    new_acetate = []
    res_atoms = residues[acetic_acid.res_num].atoms
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'C1', 99999, res_atoms[0].coord, res_atoms[0].velocity, res_atoms[0].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'C2', 99999, res_atoms[1].coord, res_atoms[1].velocity, res_atoms[1].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'H1', 99999, res_atoms[2].coord, res_atoms[2].velocity, res_atoms[2].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'H2', 99999, res_atoms[3].coord, res_atoms[3].velocity, res_atoms[3].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'H3', 99999, res_atoms[4].coord, res_atoms[4].velocity, res_atoms[4].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'O1', 99999, res_atoms[5].coord, res_atoms[5].velocity, res_atoms[5].element))
    new_acetate.append(Atom(str(res_count)+'ATX', res_count, 'ATX', 'O2', 99999, res_atoms[6].coord, res_atoms[6].velocity, res_atoms[6].element))

    # Finish product
    for atom in new_acetate:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def create_ammonia(ammonium, products_toAdd, atoms_toDelete, res_count, residues):
    """Deprotonate ammonium to create a newly formed ammonia product.

    Args:
        ammonium (Atom class object): Ammonium hydrogen donor.
        products_toAdd (list): List of newly formed product, to preserve .top file ordering, 
            do at end once all exchanges facilitated so store for now.
        atoms_toDelete (list): List of reactant Atom class objects that need to be deleted
            post exchange. Done all at once, once all exchanges facilitated, so store for now.
        res_count (int): Running tally of how many residues in system for eventual .gro file creation.
        residues (list): Ordered list of Residue class objects.

    Outputs:
       products_toAdd (list): Post exchange, updated list.
       atoms_toDelete (list): Post exchange, updated list.
       res_count (int): Updated tally.
    """
    # Delete old ammonium
    for atom in residues[ammonium.res_num].atoms:
        atoms_toDelete.append(atom)    

    # Create newly formed ammonia
    res_count += 1
    new_ammonia = []
    res_atoms = residues[ammonium.res_num].atoms
    new_ammonia.append(Atom(str(res_count)+'NXX', res_count, 'NXX', 'N1', 99999, res_atoms[1].coord, res_atoms[1].velocity, res_atoms[1].element))
    H_names = ['H11', 'H12', 'H13']
    H_count = 0
    for atom in res_atoms:
        if atom.element == 'H' and atom.atom_num != ammonium.atom_num:
            new_ammonia.append(Atom(str(res_count)+'NXX', res_count, 'NXX', H_names[H_count], 99999, atom.coord, atom.velocity, atom.element))
            H_count += 1

    # Finish product
    for atom in new_ammonia:
        products_toAdd.append(atom)
    return products_toAdd, atoms_toDelete, res_count


def do_exchanges(pairs, atoms, residues, proteins, positions, top_order, pairs_flag, prot_flag, system):
    """Final facilitation via the functions above for all accepted exchanges. Outputs corrected, post-exchange atoms.

    Args:
        pairs (list): List of Exchange class objects to be facilitated.
        atoms (list): Ordered list of Atom objects.
        residues (list): Ordered list of Residue objects.
        proteins (list): Ordered list of Protein objects.
        positions (np array): Ordered array of 3D coordinates of all atoms.
        top_order (list): Ordered list of residue names for .top files.
        pairs_flag (bool): Flag, set to True if an exchange is facilitated that
            is not a Grotthuss exchange as this requires .top file modification.  
        prot_flag (bool): Flag, set to True if any exchange involves protein 
            as this requires new protein .itp file generation.

    Outputs:
       atoms (list): Post exchange, updated list of Atom class objects. 
    """
    if not pairs:
        return atoms
    else:
        # Lists associated with changes in protonation states
        hydrogens_toAdd = []
        products_toAdd = []
        atoms_toDelete = []
        res_count = atoms[-1].res_num

        # Useful for determining which exchanges are with protein
        protein_resnames = ['LYS', 'ARG', 'GLU', 'ASP', 'HIS']

        # Need atom names of newly formed protein hydrogens
        new_HNames = {'LYS': 'HZ3', 
                    'LSN': 'HZ3',
                    'ARG': 'HH12', 
                    'ARGN': 'HH12',
                    'GLU': 'HE2', 
                    'ASP': 'HD2'
                    }

        # Update atoms list with new hydrogens, must be specifically placed after parent atom
        parent_HNames = {'LYS' : 'HZ2',
                        'LSN': 'HZ2', 
                        'ARG' : 'HH11', 
                        'ARGN': 'HH11',
                        'GLU' : 'OE2', 
                        'ASP' : 'OD2'
                        }

        # Add in termini info
        nterm_resIDs = []
        cterm_resIDs = []
        for protein in proteins:
            nterm_resIDs.append(protein.residues[0].res_id)
            cterm_resIDs.append(protein.residues[-1].res_id)
            new_HNames[f'{protein.residues[0].res_id}'] = 'H3'
            new_HNames[f'{protein.residues[-1].res_id}'] = 'HT2'
            parent_HNames[f'{protein.residues[0].res_id}'] = 'H2'
            parent_HNames[f'{protein.residues[-1].res_id}'] = 'OT2'
            if protein.residues[0].res_name not in protein_resnames: #N-term
                protein_resnames.append(protein.residues[0].res_name)
            if protein.residues[-1].res_name not in protein_resnames: #C-term
                protein_resnames.append(protein.residues[-1].res_name)

        # Change donor protonation state
        for pair in pairs:
            # Determine if Grotthuss first, these only require topology switching so handled differently
            if pair.donor.res_name == 'HHO' and pair.acceptor.res_name == 'SOL': #Grotthuss H3O+
                atoms = grotthuss_H3O(atoms, pair)
            elif pair.donor.res_name == 'SOL' and pair.acceptor.res_name == 'OHX': #Grotthuss OH-
                atoms = grotthuss_OH(atoms, pair)
            elif pair.donor.res_name == 'HHO': #H3O+ -> water
                products_toAdd, atoms_toDelete, res_count = create_water_from_hydronium(pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.donor.res_name == 'SOL': #water -> OH-
                products_toAdd, atoms_toDelete, res_count = create_hydroxide_from_water(pair.donor, products_toAdd, atoms_toDelete, res_count, residues) 
            elif pair.donor.res_name == 'AHX': #acetic acid -> acetate
                products_toAdd, atoms_toDelete, res_count = create_acetate(pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.donor.res_name == 'NXH': #NH4+ -> NH3
                products_toAdd, atoms_toDelete, res_count = create_ammonia(pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.donor.res_name in protein_resnames: #protein-H -> protein
                atoms, atoms_toDelete = deprotonate_prot(pair.donor, atoms, atoms_toDelete)
                
        # Change acceptor protonation state
        for pair in pairs:
            # Grotthuss/intra-protein exchanges handled in above loop so skip here
            if pair.donor.res_name == 'HHO' and pair.acceptor.res_name == 'SOL': #Grotthuss H3O+
                continue
            elif pair.donor.res_name == 'SOL' and pair.acceptor.res_name == 'OHX': #Grotthuss OH-
                continue
            elif pair.donor.res_name in protein_resnames and pair.acceptor.res_name in protein_resnames:
                continue 
            elif pair.acceptor.res_name == 'SOL': #water -> H3O+
                products_toAdd, atoms_toDelete, res_count = create_hydronium_from_water(pair.acceptor, pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.acceptor.res_name == 'OHX': #OH- -> water
                products_toAdd, atoms_toDelete, res_count = create_water_from_hydroxide(pair.acceptor, pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.acceptor.res_name == 'ATX': #acetate -> acetic acid
                # If Ac protonated at 180* will break, so correct here by finding new H location
                mid_atom = [x.coord for x in system.residues[pair.acceptor.res_num].atoms if x.atom_name == 'C2'][0]
                if abs(angle(mid_atom - pair.acceptor.coord, pair.donor.coord - pair.acceptor.coord) - 180) < 3:
                    new_Hloc = findMPM_H_loc(pair.donor, pair.acceptor, positions)
                else:
                    new_Hloc = None
                products_toAdd, atoms_toDelete, res_count = create_aceticAcid(pair.acceptor, pair.donor, products_toAdd, atoms_toDelete, res_count, residues, new_Hloc)
            elif pair.acceptor.res_name == 'NXX': #NH3 -> NH4+
                products_toAdd, atoms_toDelete, res_count = create_ammonium(pair.acceptor, pair.donor, products_toAdd, atoms_toDelete, res_count, residues)
            elif pair.acceptor.res_name in protein_resnames: #protein -> protein-H
                # If GLU/ASP protonated at 180* will break, so correct here by finding new H location
                if pair.acceptor.res_name in ['GLU', 'ASP']:
                    if pair.acceptor.res_name == 'GLU':
                        mid_atom = [x.coord for x in system.residues[pair.acceptor.res_num].atoms if x.atom_name == 'CD'][0]
                    elif pair.acceptor.res_name == 'ASP':
                        mid_atom = [x.coord for x in system.residues[pair.acceptor.res_num].atoms if x.atom_name == 'CG'][0]

                    if abs(angle(mid_atom - pair.acceptor.coord, pair.donor.coord - pair.acceptor.coord) - 180) < 3:
                        new_Hloc = findMPM_H_loc(pair.donor, pair.acceptor, positions)
                    else:
                        new_Hloc = None
                    atoms, hydrogens_toAdd = create_protH(pair.acceptor, pair.donor, atoms, hydrogens_toAdd, new_Hloc, nterm_resIDs, cterm_resIDs, new_HNames)
                else:                    
                    atoms, hydrogens_toAdd = create_protH(pair.acceptor, pair.donor, atoms, hydrogens_toAdd, None, nterm_resIDs, cterm_resIDs, new_HNames)

        ################# BEGIN UPDATING COORD/TOP FILES #################

        # Delete old reactants
        if pairs_flag:
            atoms_toDelete = sorted([atom.atom_num for atom in atoms_toDelete], reverse=True)
            for index in atoms_toDelete:
                del atoms[index]

        if prot_flag:
            # Sort the list in descending order based on atom_num
            sorted_hydrogens_toAdd = sorted(hydrogens_toAdd, key=lambda atom: atom.atom_num)
            sorted_hydrogens_toAdd.reverse()

            # HIS is multisite, so need to take that into account
            for hydrogen in sorted_hydrogens_toAdd:
                if hydrogen.res_name.find('HIS') != -1:
                    if hydrogen.atom_name == 'HE2':
                        parent_HNames[hydrogen.res_id] = 'NE2'          
                    if hydrogen.atom_name == 'HD1':
                        parent_HNames[hydrogen.res_id] = 'ND1'

            # Begin inserting new hydrogens
            len_prot = len(system.protein_atoms)+10
            len_prot = len_prot if len(system.atoms) > len_prot else len(system.atoms)
            for hydrogen in sorted_hydrogens_toAdd:
                for index, atom in enumerate(reversed(atoms[:len_prot])):
                    if atom.res_name != 'SOL' and atom.res_id == hydrogen.res_id:
                        if hydrogen.atom_name in ['H1', 'H2', 'H3', 'HT2'] or hydrogen.res_name == 'HIS':
                            if atom.atom_name == parent_HNames[hydrogen.res_id]:
                                atoms.insert(len_prot-index, hydrogen)          
                        else:
                            if atom.atom_name == parent_HNames[hydrogen.res_name]:
                                atoms.insert(len_prot-index, hydrogen)

        if pairs_flag:
            # Add in new products
            for atom in products_toAdd:
                atoms.append(atom)

            # Very specific ordering in top file or else will break
            atoms = update_atoms(atoms, top_order)
        return atoms