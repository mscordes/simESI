class Atom:
    """Defines all info related to a particular atom.

    Attributes:
        res_id (str): Residue number + residue name.
        res_num (int): Residue number.
        res_name (str): Residue name.
        atom_name (str): Atom name.
        atom_num (int): Atom number.
        coord (np array): 3D coordinate of atom.
        velocity (np array): 3D atom velocity.
        element (str): Atom element.
    """

    def __init__(self, res_id, res_num, res_name, atom_name, atom_num, coord, velocity, element):
        self.res_id = res_id
        self.res_num = res_num
        self.res_name = res_name        
        self.atom_name = atom_name
        self.atom_num = atom_num
        self.coord = coord
        self.velocity = velocity
        self.element = element
    
    def __str__(self):
        return f"Atom: {self.atom_name} Residue: {self.res_num}{self.res_name} Coordinates: {self.coord}"


class Residue:
    """Defines all info related to a particular residue, including complete atomic composition.

    Attributes:
        res_id (str): Residue number + residue name.
        res_num (int): Residue number.
        res_name (str): Residue name.
        atoms (list): Atom class objects that compose residue, ordered.
    """

    def __init__(self, res_id, res_num, res_name):
        self.res_id = res_id
        self.res_num = res_num
        self.res_name = res_name
        self.atoms = []
    
    def add_atom(self, atom):
        self.atoms.append(atom)
    
    def __str__(self):
        return f"Residue: {self.res_id}"
    

class Protein:
    """Defines all info related to a particular protein monomer, including complete atomic and residue composition.

    Attributes:
        residues (list): Resiude class objects that compose the protein, ordered.
        atoms (list): Atom class objects that compose the protein, ordered.
    """

    def __init__(self):
        self.residues = []
        self.atoms = []
    
    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_residue(self, residue):
        self.residues.append(residue)
    
    def __str__(self):
        return f"Protein: {len(self.residues)} amino acids from {self.residues[0]} to {self.residues[-1]}"


class Titratable_Sites:
    """Defines all info related to all titratable atoms of a titratable residue type for both protonation states.

    The nested Protonation_state class is called for both states. The objects below are lists as 
    a titratable residue can have multiple proton donors (ie., Lys terminal H's) or multiple
    proton acceptors (ie. either carboxyl oxygen in ASP/GLU).

    Attributes:
        prot (object): Info related to the protonated state
        deprot (object): Info related to the deprotonated state
    """

    def __init__(self, prot_atoms, prot_coords, prot_nearWaters, prot_clusters, \
                 deprot_atoms, deprot_coords, deprot_nearWaters, deprot_clusters):
        
        class Protonation_state:

            """
            Attributes:
                atoms (list): Atom class objects that compose the residue, ordered.
                coords (np array): 3D atom coordinates, ordered.
                nearWaters (list): Integers of number of coordinated waters to each atom, ordered.
                clusters (list): Integers of number of residue Id's solvating each atom, ordered.
            """

            def __init__(self, atoms, coords, nearWaters, clusters):
                self.atoms = atoms
                self.coords = coords
                self.nearWaters = nearWaters
                self.clusters = clusters

        self.prot = Protonation_state(prot_atoms, prot_coords, prot_nearWaters, prot_clusters)
        self.deprot = Protonation_state(deprot_atoms, deprot_coords, deprot_nearWaters, deprot_clusters)


class Donor_Acceptor:
    """Defines all info related to a particular atom involved in a potential proton exchange.
    
    This is similar to an atom object, but has more information related to the coming 
    energy calculation of the potenial exchange.

    Attributes:
        atom: Atom class object. 
        coord (np array): 3D coordinate of atom.
        nearwaters (int): Number of waters coordinating with atom.
        cluster (int): Cluster ID of cluster solvating atom. 
    """

    def __init__(self, atom, coord, nearWaters, cluster):
        self.atom = atom
        self.coord = coord
        self.nearWaters = nearWaters
        self.cluster = cluster

    def __str__(self):
        return f"Atom: {self.atom} Coordinates: {self.coord} Near Waters: {self.nearWaters} Cluster: {self.cluster}" 


class Exchange:
    """Defines all info related to an accepted proton exchange.

    Attributes:
        donor: Atom class object corresponding to proton donor. 
        acceptor: Atom class object corresponding to proton acceptor. 
        energy (float): Energy associated with proton transfer.
        hop (int): Since exchanges computed 5x per timestep for Grotthuss diffusion, 
             denotes which of 5 hops exchange occurs at.
        cluster (int): Denotes which cluster solvates donor-acceptor pair.
        near_waters (int): Denotes number of waters coordinated with donor-acceptor pair. 
    """

    def __init__(self, donor, acceptor, energy, hop, cluster, near_waters):
        self.donor = donor
        self.acceptor = acceptor
        self.energy = energy  
        self.hop = hop 
        self.cluster = cluster
        self.near_waters = near_waters

    def __str__(self):
        return f"Donor: {self.donor} Acceptor: {self.acceptor} Energy: {round(self.energy, 2)} Hop: {self.hop} Cluster: {self.cluster} Near Waters {self.near_waters}"


class System:
    """This object contains most information needed for other calculations.
    Attributes are widely ranging, but given the number of variables used throughout this program,
    it it advantageous to have everything in a single object.

    Attributes:
        proteins (list): Protein class objects corresponding to each monomer (if complex), ordered.
        residues (list): Ordered list of residue class objects.
        atoms (list): Ordered list of atom class objects.
        box_vectors (list): Coordinates corresponding to the simulation box size.
        positions (np array): Ordered coordinates of all atoms.
        positions (np array): Ordered velocities of all atoms.
        res_dict (dict): Dictionary of residues with resname as key ('LYS' for lysine in example) with value corresponding 
                  to an ordered list of residue objects of the key residue. 
        num_molecs (dict): Dictionary, similar to res_dict with resname keys, with values corresponding to the number of 
                    that residue present in the simulation.
        protein_atoms (list): Ordered list of atom objects, omitting all non-protein residues.
        charges (np array): Partial charges of all atoms (in same order as atoms attribute).
        prot_charge (int): Net charge of the protein .
        waterO_coords (np array): Ordered coordinates of all water oxygens.
        waterH_coords (np array): Ordered coordinates of all water hydrogens.
        waterO_atoms (list): Ordered atom objects of all water oxygens.
        waterH_atoms (list): Ordered atom objects of all water hydrogens.
        sys_charge (int): Net charge of the entire system (including protein).
    """
    
    def __init__(self, proteins, residues, atoms, box_vectors, positions, res_dict, num_molecs, protein_atoms, \
                 protein_coords, charges, prot_charge, waterO_coords, waterH_coords, waterO_atoms, waterH_atoms, sys_charge):
        self.proteins = proteins
        self.residues = residues
        self.atoms = atoms
        self.box_vectors = box_vectors
        self.positions = positions 
        self.res_dict = res_dict   
        self.num_molecs = num_molecs
        self.protein_atoms = protein_atoms
        self.protein_coords = protein_coords
        self.charges = charges
        self.prot_charge = prot_charge
        self.waterO_coords = waterO_coords
        self.waterH_coords = waterH_coords
        self.waterO_atoms = waterO_atoms
        self.waterH_atoms = waterH_atoms
        self.sys_charge = sys_charge