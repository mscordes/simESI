""" 
Droplet equilibriation and get pKa values.
"""
import subprocess
import os
from coordinates import unpack_pdb, write_pdb, set_protonMap, unpack_gro, write_top
from gmx import auto_gmx_input, call_mdrun, modify_tc_grps, modify_ndx_grps
from droplet_formation import form_droplet

def get_pkaVals(args):
    """Creates .pka file for a given inputted protein .pdb, then parses and returns dict of amino acid specific pka values.

    Args:
        args: argparse object of user defined args.

    Outputs:
        Dict of pKa values with keys corresponding to a titratable amino acids
    """
    #unpacks propka .pka file and returns dict of residues/pka vals
    def unpack_pkaFile(filename):
        with open(filename) as f:
            raw = [line.split() for line in f.readlines()]

        res = []
        pka = []
        for line in raw:
            if len(line) == 5:
                try:
                    res.append(int(line[1])-1)
                    pka.append(float(line[3]))
                except ValueError:
                    pass
        return dict(zip(res, pka))

    # Can set different pdb file to generate pka vals with, useful if propka3 being picky with modified coord file
    if args.pka_pdb is None: 
        coord_file = args.pdb
    else:
        coord_file = args.pka_pdb

    proteins, _, _, box_vectors, _, _, _, temp_factors = unpack_pdb(f'{coord_file}')
    write_pdb(f'{coord_file}', proteins, box_vectors, temp_factors)
    subprocess.run(f'python -m propka {coord_file}', shell=True)
    try:
        pka_vals = unpack_pkaFile(f'{coord_file[0:-4]}.pka')
    except FileNotFoundError:
        print("\n\n\nERROR: PROPKA .pka file generation failed. Check that the inputted .pdb file is properly formatted."
              "A common issue is that the final oxygen of the C-termini for each monomer must have the residue name 'OXT'" 
              "not 'OT1' or 'OT2'.")
        exit(1)
    return pka_vals 


def equilibriate(args):
    """
    Cleans up inputted protein .pdb, forms droplet and then equilibriatse with 
    energy minimization followed by NVT. Also gets pKa values. 

    Args:
        args: argparse object of user defined args.  

    Outputs:
        If equilibriating, creates equilibriated droplet coordinate file (nvt.gro) 
            and molecular topology (prot.top).
        pka_vals (dict): Dict with keys corresponding to a titratable amino acids
             residue number, and values corresponding to its pKa as computed via PROPKA.
        top_order (list): Ordered list of residue names for .top files.
    """ 

    # Gets .pka file with residue specific pka values, important for energy calcs later
    pka_vals = get_pkaVals(args)

    # Gromacs .top files require consistent residue ordering
    top_order = ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH', 'NNN', 'OOO']

    # Form droplet and equilibriate unless continuation of run, then skip
    if args.equil:
        print('Beginning equilibriation.')
        
        # Sanitize .pdb, we need consistent chain and residue numbering 
        proteins, _, _, box_vectors,_, _, _, temp_factors = unpack_pdb(f'{args.pdb}')
        write_pdb('init.pdb', proteins, box_vectors, temp_factors)
        print('Resiude & chain numbering fixed.')

        # Strip non-protein residues from file
        auto_gmx_input('gmx make_ndx -f init.pdb -o init.ndx', ['q'], args)
        auto_gmx_input('gmx editconf -f init.pdb -o init.pdb -n init.ndx -box 15 -c', [1, 1], args)
        print('Contaminants/waters removed from protein structure.')

        # Check if coord file can form valid topology, PRO and MET need alternate inputs so check here
        pdb2gmx_inputs = []
        for protein in proteins:
            # Nterm input 
            if protein.residues[0].res_name in ['PRO', 'MET']:
                pdb2gmx_inputs.append(1)
            else:
                pdb2gmx_inputs.append(0)
            
            #Cterm input always 0
            pdb2gmx_inputs.append(0)
        auto_gmx_input(f'gmx pdb2gmx -f init.pdb -o init.gro -p temp.top -ff charmm36 -water none -ignh -ter', pdb2gmx_inputs, args)

        # Check topology
        try:
            with open('temp.top', 'r') as f:
                if not len(f.readlines()) > 1:
                    print("Inputted pdb file is compromised in some way, unable to generate topology.\n\
                            Run with --verbose set to 'yes' and see gmx pdbgmx output, then edit pdb accordingly.")
                    exit(1)
        except FileNotFoundError:
            print("\n\n\nERROR: Could not form molecular topology. This is likely to due to not having " +
                  "the GROMACS executable 'gmx' callable from command line.")
            exit(1)         

        # Create initial topology and set protonation states probabilisitically given Hend. Hass. at pH 7
        proteins, residues, atoms, box_vectors, positions, res_dict, num_molecs = unpack_gro('init.gro')
        proton_map = set_protonMap(res_dict, proteins, pka_vals)
        write_top(proteins, 'prot.top', top_order, num_molecs, proton_map, args, box_vectors, True)

        # Unpacks gro file and extracts residues & atoms after write_top() sets protonation states
        proteins, residues, atoms, box_vectors, positions, _, _ = unpack_gro('prot.gro')

        # Form droplet
        residues, atoms, num_molecs = form_droplet(atoms, residues, positions, box_vectors, args, top_order, proteins)

        # Energy minimization
        print('Starting energy minimization.')
        command = 'gmx grompp -f em.mdp -c droplet.gro -p prot.top -o em.tpr -maxwarn 200'
        auto_gmx_input(command, None, args) 
        call_mdrun('em', args)
        print('Energy minimization completed.')

        # tc-grps in nvt.mdp will break unless the specified residue name is present
        modify_tc_grps('nvt.mdp', args.init_temp, num_molecs, args)

        # NVT equilibriation
        print('Starting NVT equilibriation.')
        modify_ndx_grps('em', 'pre_nvt', num_molecs, args)
        command = 'gmx grompp -f nvt.mdp -c em.gro -p prot.top -n pre_nvt.ndx -o nvt.tpr -maxwarn 200'
        if args.verbose:
            subprocess.run(command, shell=True)
        else:
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        call_mdrun('nvt', args)  
        print('NVT equilibriation completed.')

        # Check if equilibriation completed
        if not os.path.exists('nvt.gro'):
            print('\n\n\nERROR: Equilibriation failed. This is likely due to improperly formatted inputted .pdb file. '
                  'Rerun simESI.py with --verbose set to "yes" and examine gmx output, then modify .pdb accordingly.')
            exit(1)
    return pka_vals, top_order