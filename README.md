# high-performance-computing

This is a repository of three course projects in High Performance Computing Course. The following is brief introduction to the three projects. 

### 1. MPI project: to do parallel programming on a cluster using MPI. 

The prgram is to write a parallel implementation of a program to simulate the game of Life.

The game of life simulates simple cellular automata. The game is played on a rectangular board containing cells. At the start, some of the cells are occupied, the rest are empty. The game consists of constructing successive generations of the board. The rules for constructing the next generation from the previous one are:

-death: cells with 0,1,4,5,6,7, or 8 neighbors die (0,1 of loneliness and 4-8 of over population)
-survival: cells with 2 or 3 neighbors survive to the next generation.
-birth: an unoccupied cell with 3 neighbors becomes occupied in the next generation.

For this project the game board has finite size. The x-axis starts at 0 and ends at X_limit-1 (supplied on the command line). Likewise, the y-axis start at 0 and ends at Y_limit-1 (supplied on the command line).

Hardware: the cluster has 20 cores/processors per node; for a fixed number of processes, more nodes means fewer processors utilized per node
 
Reference: http://www.cs.umd.edu/class/spring2015/cmsc714/Projects/mpi-project.html
 
 ### 2. OpenMP project: write OpenMP programs.
 
Start with a working serial program (quake.c) that models an earthquake and add OpenMP directives to create a parallel program.

Reference: http://www.cs.umd.edu/class/spring2015/cmsc714/Projects/OpenMP-project.html
