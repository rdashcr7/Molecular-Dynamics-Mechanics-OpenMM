"""
Created on Tue Dec 24 19:25:54 2024

@author: RDASH
"""

#%% This code can be used to fix pdb files where certain residue information, hydrogen atoms or other missing atoms might be missing. This is important before performing molecular dynamics simulations.

from pdbfixer import PDBFixer
from openmm.app import PDBFile
 
fixer = PDBFixer(filename="file.pdb")

# Find missing residues and add missing atoms
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

with open("file_fixed.pdb", "w") as output:
    PDBFile.writeFile(fixer.topology, fixer.positions, output)


with open('file.pdb', 'r') as infile, open('cleaned_file.pdb', 'w') as outfile:
    for line in infile:
        if not line.startswith("HETATM") or "MPD" not in line:
            outfile.write(line)


