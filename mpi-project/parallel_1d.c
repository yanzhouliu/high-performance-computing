#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"

#define MSG_TAG 0



int main(int argc, char **argv) { 

	int rank, size;
	int tag = MSG_TAG;
	MPI_Status status;
	int xlim, ylim, gen;
	int xlocal, ylocal;
	int col, row;
	int row_index;
	int **board;
	int **buffer2, **buffer1, **livenum;  /* Buffer1 store live/dead; Buffer2 store # of neighbours*/
	int *temp1, *temp2, *temp3;
	int i,j,k;
	int x,y;
	int divider, residue, num;
	FILE *fp;
	char file[10];
	int rc;
	MPI_Request request;
	double start, end;
	
	/* Initialize MPI*/
	rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS) {
     printf ("Error starting MPI program. Terminating.\n");
     MPI_Finalize();
			exit(-1);
   }
	
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	//MPI_Barrier(MPI_COMM_WORLD);
	//start = MPI_Wtime();	
	
	gen = atoi (argv[2]);
	xlim = atoi (argv[3]);
	ylim = atoi (argv[4]);
	
	/* Partition size computing*/
	col = xlim;
	row = (size-1-rank + ylim) / size;
	
	xlocal = col + 2;
	ylocal = row + 2;
	
	divider = ylim/size;
	residue = ylim%size;
	
	/* Partition and allocate memory*/
	
	/* Allocate memory for arrays with given xlim and ylim*/
	board = (int **)malloc ( sizeof( int *) * ylim);
	temp1 = (int *)malloc ( sizeof(int) *xlim * ylim);
	if (board == NULL){
		fprintf(stderr, "malloc for board array failed.");
		MPI_Finalize();
		exit(-1);
	}
	for (i = 0; i < ylim; i++){
		board[i] = &temp1[i*xlim];
	}


	buffer1 = (int **)malloc ( sizeof( int *) * ylocal);
	livenum = (int **)malloc ( sizeof( int *) * ylocal);
	temp1 = (int *)malloc ( sizeof(int) *xlocal * ylocal);
	temp2 = (int *)malloc ( sizeof(int) *xlocal * ylocal);
	if ( (buffer1 == NULL) || (livenum == NULL) ){
		fprintf(stderr, "rank %d : malloc for buffer or buffer2 array failed.", rank);
		MPI_Finalize();
		exit(-1);
	}
	if ( (temp1 == NULL) || (temp2 == NULL) ){
		fprintf(stderr, "rank %d : malloc for buffer or buffer2 array failed.", rank);
		MPI_Finalize();
		exit(-1);
	}
	for (i = 0; i < ylocal; i++){
		buffer1[i] = &temp1[i*xlocal];
		livenum[i] = &temp2[i*xlocal];
	}
	
	/* Board Initialization */

	for (j = 0; j<ylim; j++){
		for (k = 0; k<xlim; k++){
			board[j][k] = 0;
		}
	}
	
	for (j = 0; j<ylocal; j++){
		for (k = 0; k<xlocal; k++){
			buffer1[j][k] = 0;
		}
	}

	/* Rank 0 read all inputs*/
	if (rank == 0){
		fp = fopen(argv[1], "r");
		if (fp == NULL) {
			MPI_Finalize();
			exit (-1);
		}
		while (fscanf (fp, "%d%d", &y, &x) == 2){
			board[y][x] = 1;
		}
		fclose(fp);
	}
	
	/* Rank 0 sends inputs and others receive*/
	MPI_Bcast(&board[0][0], xlim*ylim, MPI_INT, 0, MPI_COMM_WORLD);

	/* Store data*/
	num = (rank-1) < residue ? rank : residue;
	row_index = divider*rank + num;
	for (j = 1; j<ylocal-1; j++){
		for (k = 1; k<xlocal-1; k++){
			buffer1[j][k] = board[j-1+row_index][k-1];
		}
	}
	/* Computation*/
	for (i = 0; i < gen; i++){
		/* Exchange boundary cells with neighbors */
		if (size > 1){
			if (rank % 2 == 0) { /* even ranks send first */
				if (rank != 0){
					MPI_Send (&buffer1[1][1], col, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD);
				}
				if (rank != size-1)	{
					MPI_Recv (&buffer1[row+1][1], col, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD, &status);
				}
				if (rank != size-1){
					MPI_Send (&buffer1[row][1], col, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD);
				}
				if (rank != 0){	
					MPI_Recv (&buffer1[0][1], col, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
				}
			} else { /* odd ranks recv first */
				if (rank != size-1){
					MPI_Recv (&buffer1[row+1][1], col, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD, &status);
				}
				MPI_Send (&buffer1[1][1], col, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD);
				MPI_Recv (&buffer1[0][1], col, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
				
				if (rank != size-1){	
					MPI_Send (&buffer1[row][1], col, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD);
				}
			}
		}
		/* Compute how many neighbours currently are alive*/
		for(j = 1; j < ylocal-1; j++){
			for (k = 1; k < xlocal-1; k++){
					livenum[j][k] = buffer1[j-1][k-1] + buffer1[j-1][k] + buffer1[j-1][k+1] + buffer1[j][k-1] + buffer1[j][k+1] + buffer1[j+1][k-1] + buffer1[j+1][k] + buffer1[j+1][k+1];
			}
		}
		/*Update*/
		for (j = 1; j<ylocal-1; j++){
			for (k = 1; k<xlocal-1; k++){
				if (livenum[j][k] == 3 )
					buffer1[j][k] = 1;
				else if (livenum[j][k] < 2 || livenum[j][k] > 3)
					buffer1[j][k] = 0;
			}
		}
		
				
	}
	
	
	num = (rank-1) < residue ? rank : residue;
	row_index = divider*rank + num;
	for (j = 1; j<ylocal-1; j++){
		for (k = 1; k<xlocal-1; k++){
			if (buffer1[j][k] == 1){
				printf("%d %d\n", j+row_index-1, k-1);
			}
		}
	}
	
	free(buffer1);
	free(livenum);
	free(board);

	//MPI_Barrier(MPI_COMM_WORLD); 

	//end = MPI_Wtime();
	MPI_Finalize();
	//if (rank == 0) { /* use time on master node */
	//	printf("Runtime = %lf\n", end-start);
	//}
	return 0;
	exit(0);
}