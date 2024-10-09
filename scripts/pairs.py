"""
Finds potential donor-acceptor pairs for proton exchanges for all possible donor-acceptor pair combos.
"""
import numpy as np
import itertools
from scipy.spatial.distance import cdist
import random
from energy import final_pairs
from classes import Donor_Acceptor

# Determines if exchange a Grotthuss exchange
def isGrotthuss(pair):
    if (pair.donor.res_name == 'HHO' and pair.acceptor.res_name == 'SOL') or \
        (pair.donor.res_name == 'SOL' and pair.acceptor.res_name == 'OHX'):
        return True
    else:
        return False


# Only allow single most favorable pH changing exchange per water cluster (some exchanges dont change pH like Grotthuss so always allow)
def isAllowed_exchange(pair):
    if isGrotthuss(pair):
        return True
    elif pair.donor.res_name == 'HHO' and pair.acceptor.res_name == 'OHX': #H3O+/OH- elim
        return True
    elif pair.donor.res_name not in ['SOL', 'HHO'] and pair.acceptor.res_name not in ['SOL', 'OHX']: #No pH change rxn's
        return True  
    else:
        return False 


def find_pairs(system, pka_vals, hop, cluster_ph, tit_sites, temperature, skip_coords, nonProt_pairs, prot, exchanges):
    """Determines which potential proton donor-acceptor pairs are coordinated, determines
    transfers reasonable, and then from these accepted exchanges, does fairly complex pair selection 
    to weeds out transfers to account for duplicates, rattling, etc. Outputs which exchanges will be facilitated. 

    Args:
        system: System class object.
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
            residue number, and values corresponding to its pKa as computed via PROPKA.
        hop: Since exchanges computed 5x per timestep for Grotthuss diffusion, 
             denotes which of 5 hops exchange occurs at.
        cluster_ph (dict): pH of each cluster given cluster label as key, pH as val.
        tit_sites (dict): Dict where keys are residue names, vals are Titratable_Sites objects. Can be empty
            dict if creating or previously defined tit_sites dict.
        temperature (float): System temperature in K.
        skip_coords (np array): Array of 3D coords that have already had an exchange at a previous hop.
            Named skip as we void these exchanges to prevent 'rattling' of proton between donor-acceptor.
        nonProt_pairs (bool): Flag, switch to True if a non-Grotthuss, non-protein exchange is accepted.
        prot (bool): Flag, switch to True if a protein exchangs is accepted.
        exchanges (list): List of Exchange class objects of accepted exchanges for all exchanges during
            timstep. This includes from previoushops and is distinct from the accepted_pairs output! 
            It is purely for writing exchanges to output before beginning the MD portion.
        step (int): Timestep being computed.
        args: Argparse object of user defined args.

    Outputs:
        accepted_pairs (list): List of Exchange class objects of accepted exchanges to be
            facilitated.
        skip_coords (list): Updated skip coords.
        nonProt_pairs (bool): Updated flag.
        prot (bool): Updated flag.
        exchanges (list): Updated exchange list. Once again only for output purposes.
    """

    # Protein residue names
    protein_resnames = ['LYS', 'ARG', 'GLU', 'ASP', 'HIS']
    for protein in system.proteins: #Add in termini resnames
        if protein.residues[0].res_name not in protein_resnames: #N-term
            protein_resnames.append(protein.residues[0].res_name)
        if protein.residues[-1].res_name not in protein_resnames: #C-term
            protein_resnames.append(protein.residues[-1].res_name)

    # Non-protein (titratable) resnames
    nonWater_resnames = protein_resnames + ['NXX', 'NXH', 'ATX', 'AHX']

    # Every combination of donor & acceptor
    combos = list(itertools.product(tit_sites.keys(), tit_sites.keys()))
    combos.remove(('SOL', 'SOL')) #water can't react with itself (2H2O -> H3O+ + OH- extremely unfavorable)

    # Find potential exchanges
    potential_pairs = []
    for combo in combos:
        donors = tit_sites[f'{combo[0]}'].prot
        acceptors = tit_sites[f'{combo[1]}'].deprot

        # Check that their are actual donor-acceptor pairs
        if len(acceptors.atoms) > 0 and len(donors.atoms) > 0:

            # Reactions with residue & solvent only computed on first hop
            if (acceptors.atoms[0].res_name == 'SOL' or donors.atoms[0].res_name == 'SOL') and \
                (acceptors.atoms[0].res_name in nonWater_resnames or donors.atoms[0].res_name in nonWater_resnames) and hop != 0:
                continue
            
            # Intramolecular protein proton transfers
            elif (acceptors.atoms[0].res_name in protein_resnames and donors.atoms[0].res_name in protein_resnames):
                continue
            
            else:
                # Calculate distances and apply cutoff of 0.25 nm
                dists = cdist(donors.coords, acceptors.coords)
                min_idxs = np.argmin(dists, axis=1)
                min_dists = dists[np.arange(len(dists)), min_idxs]
                mask = min_dists < 0.25

                # Pairs that pass threshold
                passed_accs = min_idxs[mask]
                passed_dons = np.arange(len(donors.atoms))[mask]

                # Bin into Donor_Acceptor objects and create list of final pairs for energy analysis
                pot_donors = [Donor_Acceptor(donors.atoms[donor], donors.coords[donor], 
                                donors.nearWaters[donor], donors.clusters[donor]) for donor in passed_dons]
                pot_acceptors = [Donor_Acceptor(acceptors.atoms[acceptor], acceptors.coords[acceptor], 
                                acceptors.nearWaters[acceptor], acceptors.clusters[acceptor]) for acceptor in passed_accs]
                pot_pairs = list(zip(pot_donors, pot_acceptors))

                # Add to master list
                potential_pairs += pot_pairs

    # Find accepted exchanges based on energetic favoribility
    accepted_pairs = final_pairs(potential_pairs, system, pka_vals, hop, cluster_ph, temperature, protein_resnames)

    # Prevents rattling of exchange between donor-acceptor pair
    pairs_toDelete = []
    for index, pair in enumerate(accepted_pairs):
        try:
            if np.min(np.linalg.norm(np.subtract(skip_coords, pair.acceptor.coord), axis=-1)) < 0.10:
                pairs_toDelete.append(index)
        except ValueError:
            pass
    accepted_pairs = [pair for index, pair in enumerate(accepted_pairs) if index not in pairs_toDelete]

    # Selects most favorable exchange per grotthuss pair
    grot_energies = {}
    for pair in accepted_pairs:
        if isGrotthuss(pair):
            for residue in [pair.donor, pair.acceptor]:
                try:
                    if pair.energy < grot_energies[residue.res_id]:
                        grot_energies[residue.res_id] = pair.energy
                except KeyError:
                    grot_energies[residue.res_id] = pair.energy

    pairs_toDelete = []
    for index, pair in enumerate(accepted_pairs):
        if isGrotthuss(pair):
            if not (grot_energies[pair.donor.res_id] == pair.energy and grot_energies[pair.acceptor.res_id] == pair.energy):
                pairs_toDelete.append(index)
    accepted_pairs = [pair for index, pair in enumerate(accepted_pairs) if index not in pairs_toDelete]

    # Prevents duplicate reactants between different exchange reactions by selecting most energetically favorable
    energies = {}
    for pair in accepted_pairs:
        if isGrotthuss(pair): #Grotthuss use different energy calc, but delta G is 0 (since products=reactants), so set here
            pair.energy = 0.0 + random.uniform(-0.01, 0.01) #Randomization enables direct comparison if energies the same
        for residue in [pair.donor, pair.acceptor]:
            try:
                if pair.energy < energies[residue.res_id]:
                    energies[residue.res_id] = pair.energy
            except KeyError:
                energies[residue.res_id] = pair.energy

    pairs_toDelete = []
    for index, pair in enumerate(accepted_pairs):
        if not (energies[pair.donor.res_id] == pair.energy and energies[pair.acceptor.res_id] == pair.energy):
            pairs_toDelete.append(index)
    accepted_pairs = [pair for index, pair in enumerate(accepted_pairs) if index not in pairs_toDelete]

    # Find most energetically favorable exchange per cluster
    cluster_energies = {}
    for pair in accepted_pairs:
        if not isAllowed_exchange(pair):
            try:
                if pair.energy < cluster_energies[pair.cluster]:
                    cluster_energies[pair.cluster] = pair.energy
            except KeyError:
                cluster_energies[pair.cluster] = pair.energy

    pairs_toDelete = []
    for index, pair in enumerate(accepted_pairs):
        if not isAllowed_exchange(pair) and cluster_energies[pair.cluster] != pair.energy:
            pairs_toDelete.append(index)
    accepted_pairs = [pair for index, pair in enumerate(accepted_pairs) if index not in pairs_toDelete]

    # Update skip list and flags
    for pair in accepted_pairs:
        skip_coords = np.vstack((skip_coords, pair.donor.coord))     
        skip_coords = np.vstack((skip_coords, pair.acceptor.coord))
        if pair.donor.res_name in protein_resnames or pair.acceptor.res_name in protein_resnames:
            nonProt_pairs = True
            prot = True
        else:
            nonProt_pairs = True

    # Write exchanges to output list
    for pair in accepted_pairs:
        exchanges.append(pair)
    return accepted_pairs, skip_coords, nonProt_pairs, prot, exchanges