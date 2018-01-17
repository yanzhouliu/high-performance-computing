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
	int col, row, temp;
	int row_index, col_index;
	int **board;
	int *temp1, *temp2;
	int **buffer3, **buffer1, **livenum;  /* Buffer1 store live/dead; Buffer2 store # of neighbours*/
	int i,j,k;
	int x,y;
	int divider_col, residue_col, divider_row, residue_row, num_row, num_col;
	FILE *fp;
	char file[10];
	int rc;
	int n, m, rank_y, rank_x;
	MPI_Request request;
	double start, end;
	
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
	
	/* Get input arguements*/
	gen = atoi (argv[2]);
	xlim = atoi (argv[3]);
	ylim = atoi (argv[4]);
	
	/* Partition size computing*/
	m = size; 
	n = 1;
	temp = 1;
	
	do{
		temp++;
		m = size%temp ? m : size/temp;
		n = size%temp ? n : temp;
	}while (m > n);
	
	/* Rank - coordinate*/
	rank_y = rank/m;
	rank_x = rank%m;
	
	/* Partition and allocate memory*/
	col = (m-1-rank_x%m + xlim) / m;
	row = (n-1-rank_y%n + ylim) / n;
	
	xlocal = col + 2;
	ylocal = row + 2;
	
	divider_row = ylim/n;
	residue_row = ylim%n;
	
	divider_col = xlim/m;
	residue_col = xlim%m;	
	
	/* Allocate memory for arrays with given xlim and ylim*/
	board = (int **)malloc ( sizeof( int *) * ylim);
	temp1 = (int *)malloc ( sizeof(int) *xlim * ylim);
	if (board == NULL || temp1 == NULL){
		fprintf(stderr, "malloc for board array failed.");
		MPI_Finalize();
		exit(-1);
	}
	
	for (i = 0; i < ylim; i++){
		board[i] = &temp1[i*xlim];
	}
	
	/* Allocate memory for arrays with given xlim and ylim*/
	buffer1 = (int **)malloc ( sizeof( int *) * ylocal);
	livenum = (int **)malloc ( sizeof( int *) * ylocal);
	
	buffer3 = (int **)malloc ( sizeof( int *) * 4);
	temp1 = (int *)malloc ( sizeof(int) *xlocal * ylocal);
	temp2 = (int *)malloc ( sizeof(int) *xlocal * ylocal);
	
	if ( (buffer3 == NULL) || (buffer1 == NULL) || (livenum == NULL) ){
		fprintf(stderr, "rank %d : malloc for buffer or buffer2 array failed.", rank);
		MPI_Finalize();
		exit(-1);
	}
	if ( (temp1 == NULL) || (temp2 == NULL)){
		fprintf(stderr, "rank %d : malloc for buffer or buffer2 array failed.", rank);
		MPI_Finalize();
		exit(-1);
	}
	for (i = 0; i < ylocal; i++){
		buffer1[i] = &temp1[i*xlocal];
		livenum[i] = &temp2[i*xlocal];
	}
	for (i = 0; i < 4; i++){
		buffer3[i] = (int *)malloc ( sizeof(int) *row);
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
	num_row = (rank_y-1) < residue_row ? rank_y : residue_row;
	row_index = divider_row*rank_y + num_row;
	num_col = (rank_x-1) < residue_col ? rank_x : residue_col;
	col_index = divider_col*rank_x + num_col;
	for (j = 1; j<ylocal-1; j++){
		for (k = 1; k<xlocal-1; k++){
			buffer1[j][k] = board[j-1+row_index][k-1+col_index];
		}
	}
	/* Computation*/
	for (i = 0; i < gen; i++){
	
		/* Exchange boundary cells with neighbours */
		/* row first*/
		if (n > 1){
			if (rank_y % 2 == 0) { /* even rank_y send first */
				if (rank_y != 0){
					MPI_Send (&buffer1[1][1], col, MPI_INT, (rank+size-m)%size, tag, MPI_COMM_WORLD);
				}
				if (rank_y != n-1)	{
					MPI_Recv (&buffer1[row+1][1], col, MPI_INT, (rank+m)%size, tag, MPI_COMM_WORLD, &status);
				}
				if (rank_y != n-1){
					MPI_Send (&buffer1[row][1], col, MPI_INT, (rank+m)%size, tag, MPI_COMM_WORLD);
				}
				if (rank_y != 0){	
					MPI_Recv (&buffer1[0][1], col, MPI_INT, (rank+size-m)%size, tag, MPI_COMM_WORLD, &status);
				}
			} else { /* odd ranks recv first */
				if (rank_y != n-1){
					MPI_Recv (&buffer1[row+1][1], col, MPI_INT, (rank+m)%size, tag, MPI_COMM_WORLD, &status);
				}
				MPI_Send (&buffer1[1][1], col, MPI_INT, (rank+size-m)%size, tag, MPI_COMM_WORLD);
				MPI_Recv (&buffer1[0][1], col, MPI_INT, (rank+size-m)%size, tag, MPI_COMM_WORLD, &status);
				
				if (rank_y != n-1){	
					MPI_Send (&buffer1[row][1], col, MPI_INT, (rank+m)%size, tag, MPI_COMM_WORLD);
				}
			}
		}
		/* Column*/
		if (m >1){
			if (rank_x % 2 == 0) { /* even rank_x sends first */
				if (rank_x != 0){
					for (j = 0; j<row; j++)
						buffer3[0][j] = buffer1[1+j][1];					
					MPI_Send (&buffer3[0][0], row, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD);
				}
				if (rank_x != m-1)	{
					MPI_Recv (&buffer3[1][0], row, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD, &status);
					for (j = 0; j<row; j++)
						buffer1[1+j][col+1] = buffer3[1][j];
				}
				if (rank_x != m-1){
					for (j = 0; j<row; j++)
						buffer3[2][j] = buffer1[1+j][col];
						MPI_Send (&buffer3[2][0], row, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD);
				}
				if (rank_x != 0){	
					MPI_Recv (&buffer3[3][0], row, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
					for (j = 0; j<row; j++)
						buffer1[1+j][0] = buffer3[3][j];
				}
			} else { /* odd ranks recv first */
				if (rank_x != m-1){
					MPI_Recv (&buffer3[0][0], row, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD, &status);
					for (j = 0; j<row; j++)
						buffer1[1+j][col+1] = buffer3[0][j];
				}
				for (j = 0; j<row; j++)
					buffer3[1][j] = buffer1[1+j][1];	
				MPI_Send (&buffer3[1][0], row, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD);
				
				MPI_Recv (&buffer3[2][0], row, MPI_INT, (rank+size-1)%size, tag, MPI_COMM_WORLD, &status);
				for (j = 0; j<row; j++)
					buffer1[1+j][0] = buffer3[2][j];
				if (rank_x != m-1){	
					for (j = 0; j<row; j++)
						buffer3[3][j] = buffer1[1+j][col];
					MPI_Send (&buffer3[3][0], row, MPI_INT, (rank+1)%size, tag, MPI_COMM_WORLD);
				}
			}		
		}
		
		/* Corner*/
		if (n >1 && m>1){
			if (rank == 0 && n > 1){
				MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, rank+m+1, tag, MPI_COMM_WORLD, &status);
				MPI_Send (&buffer1[row][col], 1, MPI_INT, rank+m+1, tag, MPI_COMM_WORLD);
			}
			if (rank == m-1 && n>1){
				MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, rank+m-1, tag, MPI_COMM_WORLD, &status);
				MPI_Send (&buffer1[row][1], 1, MPI_INT, rank+m-1, tag, MPI_COMM_WORLD);
			}
			
			if (rank == size-m && n>1){
				MPI_Send (&buffer1[1][col], 1, MPI_INT, rank-m+1, tag, MPI_COMM_WORLD);
				MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, rank-m+1, tag, MPI_COMM_WORLD, &status);
			}
			if (rank == size-1 && n>1){	
				MPI_Send (&buffer1[1][1], 1, MPI_INT, rank-m-1, tag, MPI_COMM_WORLD);
				MPI_Recv (&buffer1[0][0], 1, MPI_INT, rank-m-1, tag, MPI_COMM_WORLD, &status);
			}
			//boundary-up
			if (rank_y == 0 && rank_x >0 && rank_x < m-1){				
				MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD, &status);
				MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD, &status);
				
				MPI_Send (&buffer1[row][1], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD);
				MPI_Send (&buffer1[row][col], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD);
			}
			//boundary-down
			if (rank_y == n-1 && rank_x >0 && rank_x < m-1){
				if(rank_y%2 == 0){
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
				}else{
					
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
				}
			}
			//boundary-left
			if (rank_x == 0 && rank_y >0 && rank_y < n-1){
				if (rank_y %2 == 0){
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][col], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
				}else{
					MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][col], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD);
				}
			}
			//boundary-right
			if (rank_x == m-1 && rank_y >0 && rank_y < n-1){
				if (rank_y %2 == 0){
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][1], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
				}else {
					MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][1], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD);
				}
			}
			//inner
			if (rank_y >0 && rank_y < n-1 && rank_x >0 && rank_x < m-1){
				if (rank_y%2==0){
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][1], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[row][col], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
				}else{
		
					MPI_Recv (&buffer1[row+1][0], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[row+1][col+1], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[1][1], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[1][col], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD);
					
					MPI_Recv (&buffer1[0][0], 1, MPI_INT, (rank-m-1+size)%size, tag, MPI_COMM_WORLD, &status);
					MPI_Recv (&buffer1[0][col+1], 1, MPI_INT, (rank-m+1+size)%size, tag, MPI_COMM_WORLD, &status);
					
					MPI_Send (&buffer1[row][1], 1, MPI_INT, (rank+m-1+size)%size, tag, MPI_COMM_WORLD);
					MPI_Send (&buffer1[row][col], 1, MPI_INT, (rank+m+1+size)%size, tag, MPI_COMM_WORLD);
					
					
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
	
	num_row = (rank_y-1) < residue_row ? rank_y : residue_row;
	row_index = divider_row*rank_y + num_row;
	num_col = (rank_x-1) < residue_col ? rank_x : residue_col;
	col_index = divider_col*rank_x + num_col;
	for (j = 1; j<ylocal-1; j++){
		for (k = 1; k<xlocal-1; k++){
			if (buffer1[j][k] == 1){
				printf("%d %d\n", j+row_index-1, k+col_index-1);
			}
		}
	}
	
	free(board);
	free(buffer1);
	free(buffer3);
	free(livenum);
	
	//MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
	//end = MPI_Wtime();
	MPI_Finalize();
	//if (rank == 0) { /* use time on master node */
	//	printf("Runtime = %lf\n", end-start);
	//}
	return 0;
	exit(0);
	
	
}