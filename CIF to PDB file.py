"""
Created on Wed Dec  4 14:08:07 2024

@author: rickyanshumandash
"""

#%% This python script is used to convert CIF files into pdb files which are easier to handle while using the openMM package.
#%% You can use the AlphaFold server to generate CIF files of peptides or proteins which aren't available in PDB database.

from Bio import PDB
 
cif_file = 'input.cif'  
parser = PDB.MMCIFParser()
 
structure = parser.get_structure('protein', cif_file)
 
pdb_file = 'output_structure.pdb' 
io = PDB.PDBIO()
io.set_structure(structure)
io.save(pdb_file)

print(f"PDB file path: {pdb_file}")
