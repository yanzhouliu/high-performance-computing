#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define	MAXNAMELENGTH	50
void main(int argc, char *argv[]){
	
	char *inputfile;
	int gen;
	int xlim, ylim;
	int i, j, k;
	int **livenum;
	int **board;
	int x,y;
	FILE *fpin;
	clock_t begin, end;
	double time_spent;
	
	//begin = clock();
	/* Get arguments from command line*/
	inputfile = argv[1];
	gen = atoi ( argv[2] );
	xlim = atoi ( argv[3] );
	ylim = atoi ( argv[4] );

	/* Allocate memory for arrays with given xlim and ylim*/
	xlim = xlim+2;
	ylim = ylim+2;
	board = (int **)malloc ( sizeof( int *) * ylim);
	livenum = (int **)malloc ( sizeof( int *) * ylim);
	if ( (board == NULL) || (livenum == NULL) ){
		fprintf(stderr, "malloc for array failed.");
		exit(-1);
	}
	for (i = 0; i < ylim; i++){
		board[i] = (int *)malloc ( sizeof( int ) * xlim);
		livenum[i] = (int *)malloc ( sizeof( int ) * xlim);
		if ( (board[i] == NULL) || (livenum[i] == NULL)){
			fprintf(stderr, "malloc for array failed.");
			exit(-1);
		}
	}
	
	/* Open input and output file and read data in */
	fpin = fopen (inputfile, "r");	

	if (fpin == NULL){
		fprintf(stderr, "open input files failed.");
		exit(-1);
	}
	/*Initialization*/
	for (j = 0; j<ylim; j++){
		for (k = 0; k<xlim; k++){
			board[j][k] = 0;
			livenum[j][k] = 0;
		}
	}
	while (fscanf (fpin, "%d%d", &y, &x) == 2){
		board[y+1][x+1] = 1;
	}
	fclose(fpin);

	/* Computation*/
	for (i = 0; i < gen; i++){
		/* Compute how many neighbours currently are alive*/
		for(j = 1; j < ylim-1; j++){
			for (k = 1; k < xlim-1; k++){
				livenum[j][k] = board[j-1][k-1] + board[j-1][k] + board[j-1][k+1] + board[j][k-1] + board[j][k+1] + board[j+1][k-1] + board[j+1][k] + board[j+1][k+1];
			}
		}
		/*Update*/
		for (j = 1; j<ylim-1; j++){
			for (k = 1; k<xlim-1; k++){
				if (livenum[j][k] == 3 )
					board[j][k] = 1;
				else if (livenum[j][k] < 2 || livenum[j][k] > 3)
					board[j][k] = 0;
			}
		}
	}
	/*output*/
	for (j = 0; j<ylim; j++){
			for (k = 0; k<xlim; k++){
				if (board[j][k] == 1)
					printf("%d %d\n", j-1, k-1);
			}
	}

	//end = clock();
	//time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Runtime = %f\n", time_spent);
	return;
}