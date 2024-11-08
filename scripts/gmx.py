'''
Miscellaneous Functions related to calling gmx and modifying .mdp files.
'''
import subprocess

def call_mdrun(output, args):
    """Creates mdrun command based on resources as defined by command line args, then calls with subprocess.

    Args:
        output (str): gmx .tpr run file name, -deffnm allows omition of .tpr.
        args: argparse object of user defined args.

    Outputs:
        Will output the usual gmx mdrun files, most notably a coordinate file of {output}.gro.
    """
    if args.hpc:
        # Edit this string if you are pulling different resources or want to modify your mdrun call
        command = f'gmx mdrun -ntmpi 1 -nb gpu -deffnm {output}'        
    elif args.gpu:
        command = f'gmx mdrun -nb gpu -deffnm {output}'
    else:
        command = f'gmx mdrun -deffnm {output}'

    if args.verbose:
        subprocess.run(command, shell=True)
    else:
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   


def auto_gmx_input(command, inputs, args):
    """Calls a given gmx command as defined by the command arg.
    Optionally allows automated input of command line args to the gmx command.

    Args:
        command (str): gmx command.
        inputs (list, optional): List of command line args to feed into the gmx call, ordered.
        args: argparse object of user line args.

    Outputs:
        Will output gmx files based on whatever command is fed.
    """
    if inputs:
        proc = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for input in inputs:
            proc.stdin.write(f"{input}\n".encode())
        proc.stdin.flush()
        output, errors = proc.communicate()    
        if args.verbose:    
            [print(line) for line in output.splitlines()]
            [print(line) for line in errors.splitlines()]
            print(inputs)
    else:
        if args.verbose:
            subprocess.run(command, shell=True)
        else:
            subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def modify_ndx_grps(coord_name, ndx_name, num_molecs, args):
    """In simESI, gas set at 300K, but everything else can be set at any user defined temp. This func
    sets correct temp and corrects ndx-grps for callling gromp which enables temp differences. Also, 
    because we use tc-grps for every type of molecule in system, have to modify tc-grps based on what is (or
    is not) there or else gmx throws an error.

    Args:
        coord_name (str): Name of coordinate (.gro) file to generate .ndx file for.
        ndx_name (str): Name of outputted .ndx file.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
        args: Argparse object of user defined args.

    Outputs:
        Correct .ndx file.
    """

    # List of commands to feed into gmx make_ndx
    commands = []

    comm_grps = f'1 |'
    tc_grps = f''
    for molec in ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
        if num_molecs[molec] > 0:
            tc_grps += f' r {molec} |'
            comm_grps += f' r {molec} |'
    comm_grps = comm_grps[:-1]
    commands.append(comm_grps)

    if len(tc_grps) > 1:
        tc_grps = tc_grps[:-1]
        commands.append(tc_grps)

    if args.atm:
        commands.append('r NNN | r OOO')

    commands.append('q')
    auto_gmx_input(f'gmx make_ndx -f {coord_name}.gro -o {ndx_name}.ndx', commands, args)
    

def modify_tc_grps(mdp, new_temp, num_molecs, args):
    """In simESI, gas set at 300K, but everything else can be set at any user defined temp. This func
    sets correct droplet and atmosphere temp and corrects tc-grps in the .mdp file which enable temp differences. Also, 
    because we use tc-grps for every type of molecule in system, have to modify tc-grps based on what is (or
    is not) there or else gmx throws an error.

    Args:
        command (str): Name of .mdp file
        new_temp (float): New temp to set non-gas molecules to.
        num_molecs (dict): Dict with keys as residue names, and values corresponding to the number of 
                    that residue present in the simulation.
        args: Argparse object of user defined args.

    Outputs:
        Corrected .mdp file.
    """
    tc_grps_str = 'Protein '
    comm_grps_str = 'Protein_'
    ref_t_str = f'{new_temp} '
    tau_t_str = '100 '

    new_tc_grps = ''
    for molec in ['SOL', 'HHO', 'OHX', 'ATX', 'AHX', 'NXX', 'NXH']:
        if num_molecs[molec] > 0:
            new_tc_grps += f'{molec}_'
            comm_grps_str += f'{molec}_'
    comm_grps_str = comm_grps_str[:-1]

    if len(new_tc_grps) > 1:
        new_tc_grps = new_tc_grps[:-1]
        tc_grps_str += new_tc_grps
        ref_t_str += f'{new_temp} '
        tau_t_str += '5 '

    if args.atm:
        comm_grps_str += ' NNN_OOO '
        tc_grps_str += ' NNN_OOO '
        ref_t_str += '300 '
        tau_t_str += '5 '    
    
    with open(mdp, 'r') as f:
        data = f.readlines()
    for index, line in enumerate(data):
        if 'comm_grps' in line:
            data[index] = f'comm_grps = {comm_grps_str}\n'
        elif 'tc_grps' in line:
            data[index] = f'tc_grps = {tc_grps_str}\n'
        elif 'ref_t' in line:
            data[index] = f'ref_t = {ref_t_str}\n'
        elif 'tau_t' in line:
            data[index] = f'tau_t = {tau_t_str}\n'
    with open(mdp, 'w') as f:
        data = "".join(data)
        f.write(data)