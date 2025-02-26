![Cover Image](/assets/cover_image.jpeg)
# simESI
simESI (simulations of ESI) is a package of scripts written in entirely in python utilizing GROMACS for simulating electrospray ionization (ESI) of proteins in ammonium acetate containing droplets to form protonated or deprotonated protein ions. simESI enables this by allowing proton transfer reactions between discrete amino acids, water, Grotthuss diffuse H₃O⁺ and OH⁻, ammonium (NH₄⁺), ammonia (NH₃), acetate (CH₃COO⁻), and acetic acid (CH₃COOH). Additional models are included to enable modelling of ambient conditions. simESI handles the simulation from end-to-end including preprocessing of inputted protein coordinate file, droplet formation & equilibriation, and running of the simulation. The simulation can be readily modified in many ways from command line (see below). Installation is quite simple as everything is written in python so as long as you have the correct dependencies, and your protein ```.pdb``` file is properly formatted, simESI should work out of the box.

If using simESI, please cite the following paper.

Cordes, M.S.; Gallagher, E.S. Molecular Dynamics Simulations of Native Protein Charging in Electrosprayed Droplets with Experimentally Relevant Compositions. 2025. *Manuscript under review.* 
doi: https://doi.org/10.26434/chemrxiv-2024-smnj3


## Dependencies
* ```python >= 3.9```
* ```GROMACS```, tested on ```GROMACS 2022.3```, but should work on any fairly recent version.
  
### Python Modules
* ```numpy```
* ```propka```
* ```scipy >= 1.11.2```
* ```hdbscan```

## Running simESI
simESI is a command line program. To run simESI, simply navigate to the ```simESI-main``` directory and call ```simESI.py --pdb ubq.pdb```. This will start a droplet simulation with the default protein structure, ubiquitin (PDB: 1UBQ). simESI is designed to take an input ```.pdb``` file of the protein (must be ```.pdb```!) from the ```simESI-main/input_files``` directory with the rest of simulation being handled completely by simESI (see below for more details on input ```.pdb``` files). 

This will create an output directory ```simESI-main/output_files``` if ```simESI-main/output_files``` does not exist. simESI then creates a numbered subdirectory using the name of the inputted ```.pdb``` in ```simESI-main/output_files```, for example ```simESI-main/output_files/ubq_1``` if using the default ```ubq.pdb``` structure. If you were to run ```simESI.py --pdb ubq.pdb``` again, the new output directory would be ```simESI-main/output_files/ubq_2``` with each additional run counting up. Within each output dir (i.e. ```simESI-main/output_files/ubq_1```) there are two subdirectories, ```data``` and ```simulation```. ```data``` contains ```.txt``` files with all the high level information to expedite data analysis including information like protein charge, system charge, number of each molecule, and execution times at each timestep. ```simulation``` contains all the files from the simulation itself, namely coordinate files from each timestep, as well as files assocaiated with droplet creation/equilibriation. 

To adjust simulation parameters, simESI uses a number of command line args detailed below to modify the simulation and how the system/droplet are defined.  

## Notes
* In terms overall time to simulate, takes ~3 hours for small proteins like ubiquitin (8.6 kDa, 12.5 ns simulation time), ~4.5 hours for lysozyme (14 kDa, 15 ns simulation time), and ~15 hours for wheat agglutinin (35 kDa, 25 ns simulation time) using GPU acceleration. Performance will depend on both CPU and GPU strength. As droplets are all-interacting, Larger proteins can take **much** longer.
* *Ammonium acetate concentration has a significant effect on the final ion charge states produced by simESI. If charge state distributions produced from simESI are different then what would be expected experimentally, consider modifying the initial, droplet ammonium acetate concentration via the command line args detailed below. As a rule of thumb, higher ammonium acetate concentration produces lower charge states and vice versa.*
* **simESI cannot use conventional trajectories between steps!** This is an unfortunate byproduct of continually changing the number and types of molecules due to proton transfers. For posterity, simESI saves the coordinate (```.gro```) file outputted after each (4 ps) step, and trajectory (```.xtc```) file after every 10 steps to allow users to see how the simulation run is evolving. A script is included in this package, ```simESI-main/data_anal/movie.py``` that create frames from generated ```.gro``` files in pymol which can be stitched together in order to make a movie.
* **simESI is memory intensive!** Another unfortunate byproduct of the above point is that given the number of steps over an entire simulated run, many coordinate files are saved. A complete run with ubiquitin which is a very small protein will be a couple of GB's, larger proteins will be even more.
* In terms of forcefield choice, **simESI is only capable of running simulations with the ```charmm36``` forcefield**. Additionally, given the number of non-standard molecules, simESI must be run with the  ```charmm36``` version contained with ```simESI-main/share```.

## Command Line Arguments
* ```--pdb ``` *(Required)* Input ```.pdb``` filename, must be present in the ```simESI-main/input_files``` directory. Must be supplied. As an example, can set to ```ubq.pdb``` that is included in simESI by default.
  * **Warning** *simESI requires that a valid ```.pka``` from ```propka``` can be generated. You should be able to run ```propka``` without error, for the inputted ```.pdb```. Very often, ```propka``` will miss the C-termini which will break the simulation. To fix this, the final oxygen, for every C-terminus in the inputted ```.pdb```, must have the atom name ```OXT```.*
  * **Warning** *The inputted ```.pdb``` must be valid! Many stock ```.pdb``` are missing atoms or improperly defined so you'll need to fix them prior to using simESI! If you can do ```gmx mdrun``` and ```python -m propka``` with a given structure outside of simESI and recieve no errors, you should be good to go.*
  * **Warning** *All HETATMS including solvents or ligands will be automatically deleted.*
  * **Note** *The inputted ```.pdb``` can be a protein homo- or hetero-complex.*
  
### Droplet Definition Arguments
* ```--droplet_size``` *(Optional, float)* Set droplet size to given nm radius. Defaults to 1.8x protein radius assuming sphere with 1.22 g/cm³ density. For ubiquitin this leads to ~2.5 nm radius droplets.
* ```--esi_mode``` *(Optional)* Choose to simulate a positive-mode ESI with ```pos``` or negative-mode ESI with ```neg```. Defaults to ```pos```.
* ```--amace_conc``` *(Optional, float)* Inital molar concentration of ammonium acetate to seed droplet with. Default is 0.25.

### System Definition Arguments
* ```--end``` *(Optional)* Choose when to end run. Options are ```desolvated``` which ends run if number of waters = 0, or ```cutoff``` which ends once time cutoff as defined by ```--time```. Default is ```cutoff```.
* ```--time``` *(Optional, float)* Time in ns in which to end the simulation. Default of 12.5 ns. This works well for ubiquitin, but for larger proteins will need to increase. 
* ```--water_cutoff``` *(Optional, int)* Once cutoff reached, ramp droplet temperature to evaporate the last handful waters. Will calculate based on protein mass default. For ubiquitin this yields 30 waters. 
* ```--atm``` *(Optional)* Choose to simulate droplet in fully atomistic atmosphere of 79% N₂ and 21% O₂ at 300K if ```yes```, or hard vacuum if ```no```. Default of ```yes```.
* ```--init_temp``` *(Optional, float)* Initial droplet temperature in K. Higher temperature = faster evaporation and much faster overall simulation, but risk denaturing protein. Default of 370.
* ```--final_temp``` *(Optional, float)* Final droplet temperature in K once ```--water_cutoff``` reached. Default of 450. For no temperature ramp set to 370.

### Compute Arguments
* ```--gpu``` *(Optional)* Choose to use GPU acceleration if ```yes``` which is *highly* recommended to the length of the simulations. Can run purely on CPU if ```no```. 
 * **Warning** *simESI assumes GPU accessibility with the default of ```yes```.*
* ```--hpc``` *(Optional)* Choose to run with more ```gmx mdrun``` params. This assumes 1 node and GPU acceleration. If using different resources/setup, you may need to manually modify some ```subprocess``` calls within simESI related to ```gmx mdrun``` to reflect the resources you are pulling (modify the gmx command in line 18 in gmx.py and line 184 in simulation.py). Default of ```no```.

### Output Arguments
* ```--verbose``` *(Optional)* Choose whether or not to hide ```gmx``` outputs from terminal. **Note** ```yes``` gives an overwhelming volume of text, but very useful if troubleshooting ```gmx``` calls. Default of ```no```.
* ```--save``` *(Optional)* Choose to save all files associated with the simulation. ```gmx mdrun``` calls give a lot of associated files that we largely don't care about. simESI also writes additional files for processes that can (usually) be safely deleted. Default of ```no``` to not save, give ```yes``` to save all files. **Warning** *These files can add up quite quickly, a single ~20 ns run with ubiquitin with all files saved can be >20 GB! If excess files deleted, that drops to ~4 GB.*

### Run Continuation Arguments
*If runs fail for whatever reason, and you need to restart from specific step in a specific trial run, use the args below to continue the run. If starting from the failed step does not work, consider starting well before it.*
**Note** Also, you must supply the ```.pdb ``` file to  ```--pdb ``` that was used when the dir was created in addition to ```--dir``` and ```--step```!* 
* ```--dir``` *(Optional)* Name of subdirectory in ```simESI-main/output_files``` to continue from. For example, if using the default ```ubq``` this could be ```ubq_1``` (see **Running simESI** section for more details on directory naming).
* ```--step``` *(Optional)* Step number to continue from (ie. the step when the run failed). simESI will generate a fresh ```.top``` file, so you can restart from any given steps ```.gro``` file. **Warning** *If restarting from a step, data will automatically append to the end of the data files in the ```data``` subdirectory (see below), which may compromise data continuity. It's on the user to modify those files accordingly.*

## Data Analysis
simESI continually saves data associated from each step in the run in the data subdirectory (ie. for a default ubiquitin run this would be in, ```simESI-main/output_files/ubq_1/data```, see the **Running simESI** section for more details). Included in ```simESI-main/data_anal``` subdirectory are two example scripts that demonstrate how the high level data stored in the ```data``` directory can be analyzed detailed below. Additionally, a third file ```movie.py``` is included that produces frames from individual simESI ```.gro``` files which can be stitched together to create a movie.

1. ```system_info.py``` This script plots the evolution of droplet composition in addition to protein charge, and ratio of net droplet charge to the droplets Rayleigh limit. To choose the trial to plot, input the trial directory in ```simESI/output_files``` to ```--dir``` (ie. ```ubq_1``` for a default run with ubiquitin). Additionally, computing the Rayleigh limit requires protein mass which is given (in kDa) to the ```--mass``` argument (ie. 8.6 for ubiquitin).

<p align="center">Example output from system_info.py using ubiquitin as an example.</p>

<p align="center">
  <img src="/assets/sysinfo_ex_ubq.png" width="500">
</p>

2. ```runtime.py``` This script is useful for analyzing the performance of simESI during a given trial. To choose the trial to plot, input the trials directory in ```simESI/output_files``` to ```--dir``` (ie. ```ubq_1``` for a default run with ubiquitin). This creates four stacked plots including the execution times from each timestep of the simESI exchange script (Exchange Calculation), MD run file preparation from ```gmx grompp``` (Simulation Preparation), the actual MD simulation (MD Simulation), and the sum of the previous three (Timestep Total). Another larger plot is created at right showing the cumulative time of the simulation in hours. In general simESI is fairly well optimized. While performance is dependent on both CPU/GPU strength, in general, the calculations done by simESI take ~10-20% of total compute time. The majority of computational expense is from the actual MD simulation or ```gmx``` calls including ```grompp``` or ```pdb2gmx```.
 
<p align="center">Example output from runtime.py using ubiquitin as an example.</p>

![runtime.py Example](/assets/runtime_ex_ubq.png) 
