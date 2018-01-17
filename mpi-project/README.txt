Here are files description:

Project Report_Yanzhou_Liu.pdf		project report

sequential.c		sequential code for life-of-game
parallel_1d.c		parallel code with 1D-decomposition for life-of-game
parallel_2d.c		parallel code with 2D-decomposition for life-of-game

makeme_seq.sh		compile sequential code; filename in the usage is without extension ".c"; usage: makeme_seq.sh <filename>
makeme_1d.sh		compile parallel with 1D-decomposition code; filename in the usage is without extension ".c"; usage: makeme_1d.sh <filename>
makeme_2d.sh		compile parallel with 2D-decomposition code; filename in the usage is without extension ".c"; usage: makeme_2d.sh <filename>

runme_seq.sh		sequential code's shell file for submitting job
runme_1d.sh			parallel with 1D-decomposition code's shell file for submitting job
runme_2d.sh			parallel with 2D-decomposition code's shell file for submitting job

All parallel code runme scripts have configuration with 32 cores, 2 nodes and 16 cores per node.

Performance results are in the project report.