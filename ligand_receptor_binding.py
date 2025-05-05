"""
Created on Mon Mar 17 12:52:47 2025

@author: rdash
"""
#%% This code can be used to a) perform molecular dyanamics of a ligand (pdb format file), b) molecular dynamics run of the receptor in pdb format c) form a complex of equilibrated ligand and receptor and d) perform molecular dynamics run of complex
#%% Remember to have the Amber14 forcefield folder downlaoded in the same directory as this script.
#%% Adjust the timestep, number of steps and sample interval to set the simulation time (25 ns in this code).
#%% This code also computes various energy terms, Radius of gyration, end to end distance and Root Mean Square DEviation from the intial structure (ligand only in complex). 
#%% For running in high performance clusters, slight tweaking might be needed such as not importin IPython. 

#%% Import all libraries

import IPython as IP
IP.get_ipython().magic('reset -sf')
import numpy as np
import os
from openmm.app import PDBFile, ForceField, Modeller, Simulation, PME
from openmm import unit, Vec3, MonteCarloBarostat, LangevinIntegrator,LangevinMiddleIntegrator
from openmmtools.integrators import VVVRIntegrator
from pdbfixer import PDBFixer
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
from simtk.unit import nanometer
from openmm.app import PDBFile, Modeller
import pandas as pd
from openmm import NonbondedForce
from openmm.app import *
from openmm import *
#from openmm.unit import *
from sys import stdout
from scipy.spatial.transform import Rotation as R
import math
from openmmtools.alchemy import AbsoluteAlchemicalFactory, AlchemicalRegion
from openmmtools import states, mcmc, multistate
from openmmtools.constants import kB
from openmmtools.multistate import MultiStateReporter, ReplicaExchangeAnalyzer
from openmm import unit
from simtk.openmm import LangevinIntegrator, LocalEnergyMinimizer
from simtk.openmm.app import Simulation

    
#%% Enter the ligand file, the receptor file, the forcefield file and the water model files

ligand_file = "Ligand.pdb"
receptor_file = "Receptor.pdb"
forcefield_file = "Forcefield_files/amber14-all.xml"
water_model = "Forcefield_files/amber14/tip3pfb.xml"

#%% Fix the PDB files

# fix Ligand
ligand_fixer = PDBFixer(ligand_file)
ligand_fixer.findMissingResidues()
ligand_fixer.findMissingAtoms()
ligand_fixer.addMissingAtoms()
ligand_fixer.addMissingHydrogens(7.0)

# save initial ligand structure
temp_ligand_initial = "initial_ligand.pdb"
with open(temp_ligand_initial, "w") as output:
    PDBFile.writeFile(ligand_fixer.topology, ligand_fixer.positions, output)

# fix Receptor
receptor_fixer = PDBFixer(receptor_file)
receptor_fixer.findMissingResidues()
receptor_fixer.findMissingAtoms()
receptor_fixer.addMissingAtoms()
receptor_fixer.addMissingHydrogens(7.0)

# save initial receptor structure
temp_receptor_initial = "initial_receptor.pdb"
with open(temp_receptor_initial,"w") as output:
    PDBFile.writeFile(receptor_fixer.topology, receptor_fixer.positions, output)
    

#%% Enter the Simulation parameters

temperature = 298.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
collision_rate = 1.0 / unit.picoseconds
timestep = 1.0 * unit.femtoseconds  
nvt_steps = 2500000
npt_steps = 100000000
max_Iter = 1000
sample_interval = 2500
n_samples = nvt_steps//sample_interval

#%% create systems

forcefield = ForceField(forcefield_file, water_model)


# prepare ligand system
ligand_pdb = PDBFile("initial_ligand.pdb")
ligand_modeller = Modeller(ligand_pdb.topology, ligand_pdb.positions)

# displace atom positions slightly
ligand_positions = np.array(ligand_modeller.positions.value_in_unit((nanometer)))

# magnitude of displacement
scale = 0.1  # 0.1 nanometers = 1 Å
noise = np.random.normal(scale=scale, size=ligand_positions.shape)

# Apply the random displacement
ligand_positions_randomized = ligand_positions + noise

ligand_modeller.positions = ligand_positions_randomized * unit.nanometers

ligand_modeller.addSolvent(forcefield, model='tip3p', 
                        ionicStrength=0.15*unit.molar,
                        neutralize=True,
                        padding=1.0*unit.nanometer
                       )


# incorporate forcefield in the ligand system
ligand_system = forcefield.createSystem(ligand_modeller.topology, nonbondedMethod=LJPME,
                                    nonbondedCutoff= 1*unit.nanometer,removeCMMotion=True,constraints=AllBonds
                                    )

# prepare receptor system
receptor_pdb = PDBFile("initial_receptor.pdb")
receptor_modeller = Modeller(receptor_pdb.topology, receptor_pdb.positions)
receptor_modeller.addSolvent(forcefield, model='tip3p', 
                        ionicStrength=0.15*unit.molar,
                        neutralize=True,
                        padding=1.0*unit.nanometer
                       )

# incorporate forcefiled in the receptor system
receptor_system = forcefield.createSystem(receptor_modeller.topology, nonbondedMethod=LJPME,
                                    nonbondedCutoff= 1*unit.nanometer,removeCMMotion=True,constraints=AllBonds
                                    )

#%% Minimize ligand with constrained bonds

ligand_integrator = LangevinMiddleIntegrator(temperature,collision_rate,timestep)
ligand_simulation = Simulation(ligand_modeller.topology, ligand_system, ligand_integrator)
ligand_simulation.context.setPositions(ligand_modeller.positions)
ligand_init_state = ligand_simulation.context.getState(getEnergy=True)
ligand_init_PE = ligand_init_state.getPotentialEnergy()

ligand_simulation.minimizeEnergy(tolerance=1*unit.kilojoules_per_mole/unit.nanometer, maxIterations=max_Iter) # LBFGS algorithm
ligand_second_state = ligand_simulation.context.getState(getEnergy=True)
ligand_second_PE = ligand_second_state.getPotentialEnergy()


# save ligand after geometry optimization
temp_minimized_ligand_file = f"minimized_ligand.pdb"
ligand_positions = ligand_simulation.context.getState(getPositions=True).getPositions()
with open(temp_minimized_ligand_file, "w") as output:
    PDBFile.writeFile(ligand_modeller.topology, ligand_positions,output)
    
def radius_of_gyration(atoms):
    """Calculate the radius of gyration for a set of atoms."""
    positions = atoms.positions
    center_of_mass = atoms.center_of_mass()
    squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1)
    rg = np.sqrt(np.mean(squared_distances))
    return rg


#%% Minimize ligand again with NO bond constraints

# re-create system with constraints=None
ligand_system_unconstrained = forcefield.createSystem(ligand_modeller.topology, nonbondedMethod=LJPME,
                                    nonbondedCutoff= 1*unit.nanometer, removeCMMotion=False, constraints=None
                                    )

ligand_integrator_unconstrained = LangevinMiddleIntegrator(temperature,collision_rate,timestep)
ligand_simulation_unconstrained = Simulation(ligand_modeller.topology, ligand_system_unconstrained, ligand_integrator_unconstrained)
ligand_simulation_unconstrained.context.setPositions(ligand_positions)
ligand_simulation_unconstrained.minimizeEnergy(tolerance=1*unit.kilojoules_per_mole/unit.nanometer, maxIterations=max_Iter)

# save ligand after unconstrained minimization
temp_minimized_ligand_file_unconstrained = f"minimized_ligand_unconstrained.pdb"
ligand_positions_unconstrained = ligand_simulation_unconstrained.context.getState(getPositions=True).getPositions()
with open(temp_minimized_ligand_file_unconstrained, "w") as output:
    PDBFile.writeFile(ligand_modeller.topology, ligand_positions_unconstrained, output)

#%% NVT run of ligand system

for i, f in enumerate(ligand_system_unconstrained.getForces()):
    f.setForceGroup(i)
    

ligand_integrator_nvt = LangevinMiddleIntegrator(temperature, collision_rate, timestep)
ligand_simulation = Simulation(
    ligand_modeller.topology, 
    ligand_system_unconstrained, 
    ligand_integrator_nvt
)
ligand_simulation.context.setPositions(ligand_positions_unconstrained)

u1 = mda.Universe("minimized_ligand_unconstrained.pdb")
ca1 = u1.select_atoms("name CA")

ligand_time_points = []
ligand_time_ns = []
ligand_rmsd_values = []
ligand_rg_values = []
ligand_ete_values = []
ligand_energy_values = []
ligand_bond_values = []
ligand_angle_values = []
ligand_torsion_values = []
ligand_nonbonded_values = []

for i in range(n_samples):
    ligand_simulation.step(sample_interval)

    # Save positions
    temp_file = "temp_sample.pdb"
    ligand_positions = ligand_simulation.context.getState(getPositions=True).getPositions()
    with open(temp_file, "w") as output:
        PDBFile.writeFile(ligand_modeller.topology, ligand_positions, output)

    u_current = mda.Universe(temp_file)
    ca_current = u_current.select_atoms("name CA")

    current_rmsd = rms.rmsd(ca1.positions, ca_current.positions, center=True, superposition=True)
    current_rg = radius_of_gyration(ca_current)
    current_ete = math.dist(ca_current.positions[0],ca_current.positions[19])
    current_energy = ligand_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    bond_energy = ligand_simulation.context.getState(
        getEnergy=True, 
        groups={0}
    ).getPotentialEnergy()
    
    nonbonded_energy = ligand_simulation.context.getState(
        getEnergy=True, 
        groups={1}
    ).getPotentialEnergy()
    
    torsion_energy = ligand_simulation.context.getState(
        getEnergy=True, 
        groups={2}
    ).getPotentialEnergy()
    
    angle_energy = ligand_simulation.context.getState(
        getEnergy=True, 
        groups={3}
    ).getPotentialEnergy()

    current_step = (i + 1) * sample_interval
    current_time = current_step*(10**(-6))
    
    ligand_time_points.append(current_step)
    ligand_time_ns.append(current_time) 
    ligand_rmsd_values.append(current_rmsd)
    ligand_rg_values.append(current_rg)
    ligand_ete_values.append(current_ete)
    ligand_energy_values.append(current_energy)
    ligand_bond_values.append(bond_energy)
    ligand_nonbonded_values.append(nonbonded_energy)
    ligand_torsion_values.append(torsion_energy)
    ligand_angle_values.append(angle_energy)

ligand_df = pd.DataFrame({
    "Time (ns)":ligand_time_ns,
    "Step": ligand_time_points,
    "RMSD (Å)": ligand_rmsd_values,
    "Rg (Å)": ligand_rg_values,
    "End to End Distance (Å)": ligand_ete_values,
    "Potential Energy (kJ/mol)": ligand_energy_values,
    "Bond Energy (kJ/mol)": ligand_bond_values,
    "Angle Energy (kJ/mol)": ligand_angle_values,
    "Torsion Energy (kJ/mol)": ligand_torsion_values,
    "Nonbonded Energy (kJ/mol)": ligand_nonbonded_values
})

ligand_df.to_csv("ligand_analysis.csv", index=False)

# Save structure
with open("NVT_ligand.pdb", "w") as output:
    PDBFile.writeFile(ligand_modeller.topology, ligand_positions, output)

#%% Minimize receptor

receptor_integrator = LangevinMiddleIntegrator(temperature,collision_rate,timestep)
receptor_simulation = Simulation(receptor_modeller.topology, receptor_system, receptor_integrator)
receptor_simulation.context.setPositions(receptor_modeller.positions)
receptor_init_state = receptor_simulation.context.getState(getEnergy=True)
receptor_init_PE = receptor_init_state.getPotentialEnergy()


# perform geometry optimization
receptor_simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole/unit.nanometer, maxIterations=max_Iter) #LBFGS algorithm
receptor_second_state = receptor_simulation.context.getState(getEnergy=True)
receptor_second_PE = receptor_second_state.getPotentialEnergy()


# save receptor after geometry optimization
temp_minimized_receptor_file = f"minimized_receptor.pdb"
receptor_positions = receptor_simulation.context.getState(getPositions=True).getPositions()
with open(temp_minimized_receptor_file, "w") as output:
    PDBFile.writeFile(receptor_modeller.topology, receptor_positions,output)


#%% NVT run of receptor system

for i, f in enumerate(receptor_system.getForces()):
    f.setForceGroup(i)
    

receptor_integrator_nvt = LangevinMiddleIntegrator(temperature, collision_rate, timestep)
receptor_simulation = Simulation(
    receptor_modeller.topology, 
    receptor_system, 
    receptor_integrator_nvt
)
receptor_simulation.context.setPositions(receptor_positions)

u1 = mda.Universe("minimized_receptor.pdb")
ca1 = u1.select_atoms("name CA")

receptor_time_points = []
receptor_time_ns = []
receptor_rmsd_values = []
receptor_rg_values = []
receptor_ete_values =[]
receptor_energy_values = []
receptor_bond_values = []
receptor_angle_values = []
receptor_torsion_values = []
receptor_nonbonded_values = []

for i in range(n_samples):
    receptor_simulation.step(sample_interval)

    # Save positions
    temp_file = "temp_sample.pdb"
    receptor_positions = receptor_simulation.context.getState(getPositions=True).getPositions()
    with open(temp_file, "w") as output:
        PDBFile.writeFile(receptor_modeller.topology, receptor_positions, output)

    u_current = mda.Universe(temp_file)
    ca_current = u_current.select_atoms("name CA")

    current_rmsd = rms.rmsd(ca1.positions, ca_current.positions, center=True, superposition=True)
    current_rg = radius_of_gyration(ca_current)
    current_ete = math.dist(ca_current.positions[0],ca_current.positions[19])
    current_energy = receptor_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    bond_energy = receptor_simulation.context.getState(
        getEnergy=True, 
        groups={0}
    ).getPotentialEnergy()
    
    nonbonded_energy = receptor_simulation.context.getState(
        getEnergy=True, 
        groups={1}
    ).getPotentialEnergy()
    
    torsion_energy = receptor_simulation.context.getState(
        getEnergy=True, 
        groups={2}
    ).getPotentialEnergy()
    
    angle_energy = receptor_simulation.context.getState(
        getEnergy=True, 
        groups={3}
    ).getPotentialEnergy()

    current_step = (i + 1) * sample_interval
    current_time = current_step*(10**(-6))
    
    receptor_time_points.append(current_step)
    receptor_time_ns.append(current_time)
    receptor_rmsd_values.append(current_rmsd)
    receptor_rg_values.append(current_rg)
    receptor_ete_values.append(current_ete)
    receptor_energy_values.append(current_energy)
    receptor_bond_values.append(bond_energy)
    receptor_nonbonded_values.append(nonbonded_energy)
    receptor_torsion_values.append(torsion_energy)
    receptor_angle_values.append(angle_energy)

receptor_df = pd.DataFrame({
    "Time (ns)":receptor_time_ns,
    "Step": receptor_time_points,
    "RMSD (Å)": receptor_rmsd_values,
    "Rg (Å)": receptor_rg_values,
    "End to End Distance (Å)": receptor_ete_values,
    "Potential Energy (kJ/mol)": receptor_energy_values,
    "Bond Energy (kJ/mol)": receptor_bond_values,
    "Angle Energy (kJ/mol)": receptor_angle_values,
    "Torsion Energy (kJ/mol)": receptor_torsion_values,
    "Nonbonded Energy (kJ/mol)": receptor_nonbonded_values
})


receptor_df.to_csv("receptor_analysis.csv", index=False)

# Save structure
with open("NVT_receptor.pdb", "w") as output:
    PDBFile.writeFile(receptor_modeller.topology, receptor_positions, output)


#%% Preparing the complex

# Load receptor and ligand after NVT simulation
receptor_pdb = PDBFile("NVT_receptor.pdb")
ligand_pdb = PDBFile("NVT_ligand.pdb")

# Create Modeller object and combine receptor and ligand
complex_modeller = Modeller(receptor_pdb.topology, translated_receptor_positions)
complex_modeller.add(ligand_pdb.topology, translated_ligand_positions)

# Remove water molecules if present
water_residues = [res for res in complex_modeller.topology.residues() if res.name == 'HOH']
complex_modeller.delete(water_residues)
ion_residues = [res for res in complex_modeller.topology.residues() if res.name in ['NA', 'CL']]
complex_modeller.delete(ion_residues)

# Save the initial complex
with open("initial_complex.pdb", "w") as f:
    PDBFile.writeFile(complex_modeller.topology, complex_modeller.positions, f)

# Prepare the complex system
complex_modeller.addSolvent(forcefield, model='tip3p',
                            ionicStrength=0.15*unit.molar,
                            neutralize=True,
                   #         boxSize = Vec3(10.0,10.0,10.0)*nanometers,
                            padding=1.0*nanometer
                  )

# Create the system with specified nonbonded method
complex_system = forcefield.createSystem(complex_modeller.topology,
                                         nonbondedMethod=PME,
                                         nonbondedCutoff=1.0*nanometer,
                                         removeCMMotion=True)

#%% Minimize complex

complex_integrator = LangevinMiddleIntegrator(temperature,collision_rate,timestep)
complex_simulation = Simulation(complex_modeller.topology, complex_system, complex_integrator)
complex_simulation.context.setPositions(complex_modeller.positions)
complex_init_state = complex_simulation.context.getState(getEnergy=True)
complex_init_PE = complex_init_state.getPotentialEnergy()


# perform geometry optimization
complex_simulation.minimizeEnergy(tolerance=0.1*unit.kilojoules_per_mole/unit.nanometer, maxIterations=max_Iter)
complex_second_state = complex_simulation.context.getState(getEnergy=True)
complex_second_PE = complex_second_state.getPotentialEnergy()


# save complex after geometry optimization
temp_minimized_complex_file = f"minimized_complex.pdb"
complex_positions = complex_simulation.context.getState(getPositions=True).getPositions()
with open(temp_minimized_complex_file, "w") as output:
    PDBFile.writeFile(complex_modeller.topology, complex_positions,output)

#%% NVT run of complex system

for i, f in enumerate(complex_system.getForces()):
    f.setForceGroup(i)
    
complex_integrator_nvt = LangevinMiddleIntegrator(temperature, collision_rate, timestep)
complex_simulation = Simulation(
    complex_modeller.topology, 
    complex_system, 
    complex_integrator_nvt
)
complex_simulation.context.setPositions(complex_positions)

# select alpha carbons of ligand in complex
u1 = mda.Universe("initial_complex.pdb") 
ca1 = u1.select_atoms("segid B and name CA") 

complex_time_points = []
complex_time_ns = []
complex_rmsd_values = []
complex_rg_values = []
complex_ete_values = []
complex_energy_values = []
complex_bond_values = []
complex_angle_values = []
complex_torsion_values = []
complex_nonbonded_values = []
ligand_receptor_distance =[]

for i in range(n_samples):
    complex_simulation.step(sample_interval)

    # Save positions
    temp_file = "temp_sample.pdb"
    complex_positions = complex_simulation.context.getState(getPositions=True).getPositions()
    with open(temp_file, "w") as output:
        PDBFile.writeFile(complex_modeller.topology, complex_positions, output)

    u_current = mda.Universe(temp_file)
    ca_current = u_current.select_atoms("segid B and name CA")
    ca1_current = u_current.select_atoms("segid A and name CA")
    
    temp_lig_CA_pos = np.array(ca_current.positions)
    temp_lig_CA_com = np.mean(temp_lig_CA_pos, axis = 0)
    temp_rec_CA_pos = np.array(ca1_current.positions)
    temp_rec_CA_com = np.mean(temp_rec_CA_pos, axis = 0)
    
    current_dist = math.dist(temp_lig_CA_com,temp_rec_CA_com)
    
    current_rmsd = rms.rmsd(ca1.positions, ca_current.positions, center=True, superposition=True)
    current_rg = radius_of_gyration(ca_current)
    current_ete = math.dist(ca_current.positions[0],ca_current.positions[19])
    current_energy = complex_simulation.context.getState(getEnergy=True).getPotentialEnergy()
    
    bond_energy = complex_simulation.context.getState(
        getEnergy=True, 
        groups={0}
    ).getPotentialEnergy()
    
    nonbonded_energy = complex_simulation.context.getState(
        getEnergy=True, 
        groups={1}
    ).getPotentialEnergy()
    
    torsion_energy = complex_simulation.context.getState(
        getEnergy=True, 
        groups={2}
    ).getPotentialEnergy()
    
    angle_energy = complex_simulation.context.getState(
        getEnergy=True, 
        groups={3}
    ).getPotentialEnergy()

    current_step = (i + 1) * sample_interval
    current_time = current_step*(10**(-6))
    
    complex_time_points.append(current_step)
    complex_time_ns.append(current_time)
    ligand_receptor_distance.append(current_dist)
    complex_rmsd_values.append(current_rmsd)
    complex_rg_values.append(current_rg)
    complex_ete_values.append(current_ete)
    complex_energy_values.append(current_energy)
    complex_bond_values.append(bond_energy)
    complex_nonbonded_values.append(nonbonded_energy)
    complex_torsion_values.append(torsion_energy)
    complex_angle_values.append(angle_energy)

complex_df = pd.DataFrame({
    "Time (ns)":complex_time_ns,
    "Step": complex_time_points,
    "Ligand Receptor Centre of mass distance":ligand_receptor_distance,
    "RMSD (Å)": complex_rmsd_values,
    "Rg (Å)": complex_rg_values,
    "End to End Distance (Å)": complex_ete_values,
    "Potential Energy (kJ/mol)": complex_energy_values,
    "Bond Energy (kJ/mol)": complex_bond_values,
    "Angle Energy (kJ/mol)": complex_angle_values,
    "Torsion Energy (kJ/mol)": complex_torsion_values,
    "Nonbonded Energy (kJ/mol)": complex_nonbonded_values
})

complex_df.to_csv("complex_analysis.csv", index=False)

# save complex after NVT run
temp_NVT_complex_file = f"NVT_complex.pdb"
complex_positions = complex_simulation.context.getState(getPositions=True).getPositions()
with open(temp_NVT_complex_file, "w") as output:
    PDBFile.writeFile(complex_modeller.topology, complex_positions,output)
