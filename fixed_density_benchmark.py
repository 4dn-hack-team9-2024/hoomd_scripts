import os
import sys
import json
import hoomd
import codecs

import polychrom_hoomd.log as log
import polychrom_hoomd.build as build
import polychrom_hoomd.forces as forces

from polykit.generators.initial_conformations import grow_cubic


if len(sys.argv) != 5:
	print("Usage is %s system_size density md_steps gpu_id" % sys.argv[0])
	sys.exit()


size = int(sys.argv[1])
density = float(sys.argv[2])
md_steps = int(sys.argv[3])
gpu_id = int(sys.argv[4])

# Initialise HooMD on the GPU
hoomd_device = hoomd.device.GPU(gpu_ids=[gpu_id])

# Generate RNG seed
rng_seed = os.urandom(2)
rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)

print("Using entropy-harvested random seed: %d" % rng_seed)

# Initialize empty simulation object
system = hoomd.Simulation(device=hoomd_device, seed=rng_seed)
chromosome_sizes = [size]

# Initialize simulation with the appropriate box size
number_of_monomers = sum(chromosome_sizes)
L = (number_of_monomers/density) ** (1/3.)

snapshot = build.get_simulation_box(box_length=L)

# Build random, dense initial conformations
monomer_positions = grow_cubic(N=number_of_monomers, boxSize=int(L-2))

# Populate snapshot with the generated chains
build.set_chains(snapshot, monomer_positions, chromosome_sizes, monomer_type_list=['A'])

# Setup HooMD simulation object
system.create_state_from_snapshot(snapshot)

# Setup neighbor list
nl = hoomd.md.nlist.Cell(buffer=0.4)

# Read input force parameters
with open("force_dict_homopolymer.json", 'r') as dict_file:
    force_dict = json.load(dict_file)
    
# Set chromosome excluded volume
repulsion_forces = forces.get_repulsion_forces(nl, **force_dict)

# Set bonded/angular potentials
bonded_forces = forces.get_bonded_forces(**force_dict)
angular_forces = forces.get_angular_forces(**force_dict)

# Define full force_field
force_field = repulsion_forces + bonded_forces + angular_forces

# HooMD to openMM time conversion factor
t_conv = (1.67377*10**-27/(1.380649*10**-23*300))**0.5

# Initialize integrators and Langevin thermostat
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
integrator = hoomd.md.Integrator(dt=70*t_conv, methods=[langevin], forces=force_field)

# Setup simulation engine
system.operations.integrator = integrator

system.run(md_steps)
