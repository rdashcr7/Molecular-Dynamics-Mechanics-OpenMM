# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:25:54 2024

@author: RDASH
"""

from pdbfixer import PDBFixer
from openmm.app import PDBFile

# Load your PDB file
fixer = PDBFixer(filename="3bmp.pdb")

# Find missing residues and add missing atoms
fixer.findMissingResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)  # Add hydrogens at pH 7.0

# Save the fixed PDB file
with open("3bmp_fixed.pdb", "w") as output:
    PDBFile.writeFile(fixer.topology, fixer.positions, output)


with open('3bmp.pdb', 'r') as infile, open('cleaned_3bmp.pdb', 'w') as outfile:
    for line in infile:
        if not line.startswith("HETATM") or "MPD" not in line:
            outfile.write(line)


