In this repository, you will find essential scripts needed to run molecular dynamics and molecular mechanics studies for lignad-receptor docking. 
The CIF to PDB file is a starting point to generate necessary pdb files of molecules that do not exist. 
The PDB fixer file ensures that the missing residues, missing hydrogen and other atoms are fixed in the file.
The ligand receptor binding file performs a molecular dynamics run of the ligand, receptor, creates the complex and runs a molecular dynamics run of the complex while tabulating different important properties of the system at different instants. 
!!! It is extremely important that the folder containing Amber14 and tip3p water model xml files are in a directory which has the same pathas that of the script.
