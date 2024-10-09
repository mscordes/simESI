"""
Script that uses pymol to stitch together images of individual coordinate files 
produced by simESI to form a movie. Pay attention to the user defined vars
below. Make sure that 'trial_dir' is set to the simulation dir of the trial
you wish to make a movie for. Will create a 'frames' dir in the dir below the 
simulation dir. Use ffmpeg or some other movie maker to stitch the frames together.

TO USE:
Open pymol and cd to the dir containing this file (i.e., where the correct path to 
'~/simESI/data_anal'). Check 'trial_dir' and 'step' vars correctly set below, 
then type   run movie.py    into the pymol command line and hit enter.

NOTE: 
This script uses ray tracing for protein images so can take a fairly long time
if lots of frames to be generated.
"""
import os
import shutil
from timeit import default_timer as timer
from PIL import Image
from pymol import cmd

# USER DEFINED VARIABLES
trial_dir = '../output_files/ubq_1/simulation'
step = 100 #Step size between .gro files, 100*4ps = 400 ps or 0.4 ns simulation time, per frame
dist = 50 #Dist from protein to center image

# Frames will be stored in the 'frames' dir
os.chdir(trial_dir)
if os.path.exists('../frames'):
    shutil.rmtree('../frames')
os.mkdir('../frames')
cmd.delete('all')

# Initialize display parameters
cmd.bg_color("white") #background color
cmd.set("internal_gui", "0")
cmd.viewport(2500, 2500)  
cmd.set("depth_cue", "0") 
cmd.set("orthoscopic", "0") #perspective
cmd.set("specular", "0") #specular reflections
cmd.set("ray_trace_gain", "0.5") #ray trace outline thickness
cmd.set("ray_trace_fog", "1")
cmd.set("fog_start", "0.80")
cmd.set("ray_shadows", "0")

# Set initial viewpoint
cmd.load('0.gro', 'init')
cmd.center('poly', state=-1, origin=1)
cmd.zoom('center', dist)
init_view = cmd.get_view()
cmd.delete('all')

# Start getting frames
frame = 1
count = 0
while True:
    file = step*count
    count += 1
    if not os.path.exists(f'{file}.gro'):
        break
    else:
        start = timer()

        # Isolate waters
        cmd.load(f'{file}.gro', 'water')
        cmd.set_view(init_view)
        cmd.center('poly', state=-1, origin=0)
        view = cmd.get_view()
        cmd.select('solvent or resn NNN or resn OOO')
        cmd.remove('not sele')
        cmd.remove('name MW4')

        # Color waters 
        cmd.show('surface', 'solvent')
        cmd.set('transparency', '0.5', 'solvent')
        cmd.set('surface_color', 'marine')
        cmd.delete('sele')

        # Water image
        cmd.set_view(view)
        cmd.set("ray_trace_mode", "0")
        cmd.png(f'../frames/{file}_water.png', ray=0, dpi=300)
        cmd.delete('all')

        # Isolate ions and protein
        cmd.load(f'{file}.gro', 'ions')
        cmd.remove('resn OOO or resn NNN or solvent')

        # Show acetate/acetic acid as spheres and color red
        cmd.select('resn ATX or resn AHX')
        cmd.show_as('spheres', 'sele')
        cmd.select('resn ATX and (name C*) or resn AHX and (name C*)')
        cmd.color('red', 'sele')

        # Change sphere size of ammonium & acetate
        cmd.set('sphere_scale', '0.6')
        cmd.set('sphere_scale', '0.5', '(elem H*)')

        # Make protein green cartoon with grey surface
        cmd.select('rep cartoon')
        cmd.show('surface', 'poly')
        cmd.set('transparency', '0.80')
        cmd.set('surface_color', 'grey')
        cmd.set('cartoon_color', 'green')

        # Show deprot/prot amino acids as black/white spheres
        cmd.select('resn GLUH and name OE2 or resn ASPH and name OD2 or resn HISH and name ND1 or resi 76 and name HT2') #lyz
        cmd.show('spheres', 'sele')
        cmd.select('resn GLUH and name OE2 or resn ASPH and name OD2 or resn HISH and name ND1 or resi 76 and name HT2') #lyz
        cmd.color('white', 'sele')
        cmd.select('resn ARGN and name NH1 or resn LYSN and name NZ')
        cmd.show('spheres', 'sele')
        cmd.select('resn ARGN and name NH1 or resn LYSN and name NZ')
        cmd.color('black', 'sele')  

        # Ion and poly image
        cmd.set_view(view)
        cmd.set("ray_trace_mode", "1")
        cmd.png(f'../frames/{file}_poly.png', ray=1, dpi=300, width=2500, height=2500)
        cmd.delete('all')

        # Combine image layers
        background = Image.open(f'../frames/{file}_water.png')
        foreground = Image.open(f'../frames/{file}_poly.png')
        background.paste(foreground, (0, 0), foreground)
        background.save(f'../frames/{frame}.png')
        frame += 1

        # Delete layer .pngs
        os.remove(f'../frames/{file}_water.png')
        os.remove(f'../frames/{file}_poly.png')
        print(f'Frame {file} rendered in {"{:.2f}".format(float(timer()-start))} seconds.')
os.chdir('../')