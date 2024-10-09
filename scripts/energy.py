"""
Computes potential proton transfers.
""" 
import numpy as np
import math
import random
from classes import Exchange

# Buffer pKa's
nonProt_pka = { 'ATX' : 4.76, #acetate
                'AHX' : 4.76, #acetic acid
                'NXX' : 9.25, #ammonia
                'NXH' : 9.25, #ammonium
                'HHO' : 0.0,  #H3O+
                'OHX' : 14.0, #OH-
                }


"""Residue gas phase proton affinity in kJ/mol
From J. B. Klauda et al. The Journal of Physical Chemistry B 2010 Vol. 114 Issue 23 Pages 7830-7843"""
residue_PA = {  'HIS'  :  953.70,
                'LYS'  :  917.97,
                'ARG'  : 1002.10,
                'GLU'  : 1452.70,
                'ASP'  : 1452.70,
                'SOL'  : 1633.02,
                'HHO'  :  690.36,
                'NTER' :  887.00, 
                'CTER' : 1424.00,
                'ATX'  : 1452.70,
                'AHX'  : 1452.70,
                'NXX'  :  853.54,
                'NXH'  :  853.54, 
                }

def find_pkPotential(donor, acceptor, near_waters, cluster_ph, protein_resnames, pka_vals, nonProt_pka, temperature, residue_PA):
    """Computes the Delta E of a specific exchange considering the pKa of the donor and acceptor, 
    corrected to account for changes in gas phase proton affinity if waters < 30.

    Args:
        donor (Atom class object): Info for hydrogen being donated.
        acceptor (Atom class object): Info for hydrogen accepting site.
        near_waters (int): Number of waters solvating donor-acceptor pair.
        cluster_ph (dict): Dict of pH of the cluster solvating each donor-acceptor pair.
        protein_resnames (list): List of all protein residue names (i.e., LYS, ARG, GLU, etc).
            Used to determine if an exchange with protein or not.
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
             residue number, and values corresponding to its pKa as computed via PROPKA.
        nonProt_pka (dict): Dict of pKa values of non-amino acid residues. 
        temperature (float): System temperature in K.
        residue_PA (dict): Dict of residue gas phase proton affinities.

    Outputs:
        (float): Delta E of the exchange in kJ/mol. 
    """

    # Begin finding solution pKa of donor 
    if donor.atom.res_name == 'SOL':
        pka_val = cluster_ph[donor.cluster]
    elif donor.atom.res_name in protein_resnames:
        pka_val = pka_vals[donor.atom.res_num]
    else:
        pka_val = nonProt_pka[donor.atom.res_name]

    # Donor pKb
    if acceptor.atom.res_name == 'SOL':
        pkb_val = cluster_ph[acceptor.cluster]
    elif acceptor.atom.res_name in protein_resnames:
        pkb_val = pka_vals[acceptor.atom.res_num]
    else:
        pkb_val = nonProt_pka[acceptor.atom.res_name]         

    # Pure solution phase pk potential
    sol_potential = (0.019144) * (pka_val - pkb_val) * (temperature) #ln(10)*kb*Na = 0.019144 kJ/mol*K

    # If >30 near waters to residue, we assume bulk like conditions and calc takes into account only solution pka/pkb
    if near_waters > 30:
        return sol_potential
    
    # If <30 near waters, need gas phase correction
    else:
        # Set gpa
        if donor.atom.res_name in ['HHO', 'SOL']:
            # Residue gpa
            if donor.atom.res_name == 'HHO':
                gpa_val = residue_PA['HHO']
            elif donor.atom.res_name == 'SOL':
                gpa_val = residue_PA['SOL']
        else:
            # Residue gpa, but termini have to be treated specially
            if donor.atom.atom_name in ['H1', 'H2', 'H3']:
                gpa_val = residue_PA['NTER'] 
            elif donor.atom.atom_name == 'HT2':
                gpa_val = residue_PA['CTER']
            else:
                gpa_val = residue_PA[donor.atom.res_name]                  

        # Set gpb
        if acceptor.atom.res_name in ['OHX', 'SOL']:
            if acceptor.atom.res_name == 'OHX':
                gpb_val = residue_PA['SOL']
            elif acceptor.atom.res_name == 'SOL':
                gpb_val = residue_PA['HHO']
        else:
            if acceptor.atom.atom_name == 'N':
                gpb_val = residue_PA['NTER']
            elif acceptor.atom.atom_name in ['OT1', 'OT2']:
                gpb_val = residue_PA['CTER']
            else:
                gpb_val = residue_PA[acceptor.atom.res_name]

        """Employ logarithmic decay from pure gas phase proton affinities to normal pka at 30 waters
        Exponential factor fitted from A. Kumar et al. Physical Chemistry Chemical Physics 2022 Vol. 24 Issue 30 Pages 18236-18244"""
        def GP_correct(gp_val, sp_val, near_waters):
            sol_pot = 0.019144*sp_val*temperature
            return ((gp_val-sol_pot)*np.exp(-0.30312*near_waters)) + sol_pot
        return GP_correct(gpa_val, pka_val, near_waters) - GP_correct(gpb_val, pkb_val, near_waters)


def find_ES_pot(system, donor, acceptor, exchanged_charge):
    """Finds the delta of electrostatic potential for a Grotthuss exchange. This
    can be for H3O+ or OH-.

    Args:
        system (System class object)
        donor (Atom class object): Info for hydrogen being donated.
        acceptor (Atom class object): Info for hydrogen accepting site.
        exchanged_charge (float): +1.00 for H3O+, -1.00 for OH-.

    Outputs:
        (float): Delta E of the Grothuss exchange.
    """

    # Find the electrostatic energy of a single state
    def find_single_pot(exchange_atom, exchanged_charge, alt_atom):
        # ES calc relative to M location in each water
        m_atom = [atom for atom in system.residues[exchange_atom.res_num].atoms if atom.atom_name in ['O1', 'MW', 'MW4']][0]

        # Remove nearby molecules to more broadly sample electrostatic environment
        distances = np.linalg.norm(np.subtract(system.waterO_coords, m_atom.coord), axis=-1)
        del_indices = np.where(distances < 0.50)[0]
        toDelete = []
        for index in del_indices:
            for atom in system.residues[system.waterO_atoms[index].res_num].atoms:
                toDelete.append(atom.atom_num)

        # Add in exchange partners
        for atom in system.residues[exchange_atom.res_num].atoms:
            toDelete.append(atom.atom_num)
        for atom in system.residues[alt_atom.res_num].atoms:
            toDelete.append(atom.atom_num)

        # Void deleted atoms from potential calculation
        temp_positions = np.array([system.positions[x] for x in toDelete])
        for index in toDelete:
            system.positions[index] = np.inf

        # Potential of no exchange, ke = 138.932 kJ*nm/mol*e^2
        pot = (np.sum(np.divide(system.charges, np.linalg.norm(np.subtract(system.positions, m_atom.coord), axis=-1))))*138.932*(exchanged_charge)

        # Revert charges and positions, quicker than deepcopy
        for count, index in enumerate(toDelete):
            system.positions[index] = temp_positions[count]
        return pot
    
    # Compute and return the delta E between the two states
    return find_single_pot(acceptor, exchanged_charge, donor) - find_single_pot(donor, exchanged_charge, acceptor)


def MCMC(delta_e, temperature):
    """Metropolis Criterion Monte Carlo (MCMC) sampling algorithm. Accepts
    or rejects a given exchange based on Delta E and system temperature.

    Args:
        delte_e (float): Delta E of exchange in kJ/mol.
        temperature (float): System temperature in K.

    Outputs:
        (int): 1 is accepted exchange, 0 for rejection. 
    """
    prob_exchange = 1 if delta_e <= 0 else math.exp(-delta_e/(0.008314*temperature))
    return random.choices([1, 0], weights=(prob_exchange, 1-prob_exchange), k=1)[0]


def final_pairs(pair_list, system, pka_vals, hop, cluster_ph, temperature, protein_resnames):  
    """Combines the functions above and outputs accepted exchanges.

    Args:
        pair_list (list): List of coordinated proton donors and acceptors of a given type.
        system: System class object.
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
             residue number, and values corresponding to its pKa as computed via PROPKA.
        hop (int): Grotthuss diffusion can hop multiple times (up to 5 for H3O+), per timestep. 
            This determines what 'hop' the exchanges are being computed on.
        cluster_ph (dict): Dict of pH of the cluster solvating a donor-acceptor pair.
        temperature (float): System temperature in K.
        protein_resnames (list): List of all protein residue names (i.e., LYS, ARG, GLU, etc).
            Used to determine if an exchange with protein or not.

    Outputs:
        exchanges (list): List of Exchange objects accepted by the MCMC algo. 
    """
    exchanges = []
    for donor, acceptor in pair_list:
        nearWaters = donor.nearWaters if donor.nearWaters > acceptor.nearWaters else acceptor.nearWaters

        # Find potentials of exchanges, Grotthuss exchanges require electrostatics so are handled seperately
        if donor.atom.res_name == 'HHO' and acceptor.atom.res_name == 'SOL': #Grotthuss H3O+
            free_energy = find_ES_pot(system, donor.atom, acceptor.atom, 1.00)
            free_energy -= 25
            free_energy = free_energy-9999.9 if free_energy < 0 else free_energy+9999.9 #Avoid MCMC

        elif donor.atom.res_name == 'SOL' and acceptor.atom.res_name == 'OHX': #Grotthuss OH-
            free_energy = find_ES_pot(system, donor.atom, acceptor.atom, -1.00)
            free_energy = 9999.9 if hop > 2 else free_energy #OH- hops less than H3O+ so max of 3 hops instead of 5
            free_energy = free_energy-9999.9 if free_energy < 0 else free_energy+9999.9 #Avoid MCMC

        else: # All other exchanges
            free_energy = find_pkPotential(donor, acceptor, nearWaters, cluster_ph, protein_resnames, \
                                           pka_vals, nonProt_pka, temperature, residue_PA) 

        # Rand prevents exchanges having same energy which interferes with comparing (and selecting) exchanges
        free_energy += random.uniform(-0.01, 0.01)

        # Metropolis Criterion Monte Carlo (MCMC) sample free energy
        if MCMC(free_energy, temperature) == 1:
            exchange = Exchange(donor.atom, acceptor.atom, free_energy, hop, donor.cluster, nearWaters)
            exchanges.append(exchange)
    return exchanges