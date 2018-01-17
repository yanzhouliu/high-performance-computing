#!/bin/tcsh
#SBATCH --ntasks-per-node=16
#SBATCH -t 00:01:00
#SBATCH --nodes=2
module unload intel
module load openmpi/gnu

mpirun -np 32 parallel_2d final.data 500 500 500
