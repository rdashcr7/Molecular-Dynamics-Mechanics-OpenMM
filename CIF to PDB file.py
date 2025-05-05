#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:08:07 2024

@author: rickyanshumandash
"""

from Bio import PDB

# Load the CIF file
cif_file = 'fold_apoa_i_6cch_model_1.cif'  # Specify the path to your CIF file
parser = PDB.MMCIFParser()

# Parse the CIF file
structure = parser.get_structure('protein', cif_file)

# Write to PDB file
pdb_file = 'output_structure.pdb'  # Specify the desired output PDB file name
io = PDB.PDBIO()
io.set_structure(structure)
io.save(pdb_file)

print(f"PDB file saved as {pdb_file}")