#!/bin/tcsh


module unload intel
module load openmpi/gnu

rm -f $1.o
rm -f $1

module unload intel
module load openmpi/gnu

mpicc $1.c -o $1

