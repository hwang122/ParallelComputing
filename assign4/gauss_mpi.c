#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

/* Program Parameters */
#define MAXN 1000  /* Max value of N */
/* A * X = B, solve for X */
float A[MAXN][MAXN], B[MAXN], X[MAXN];


/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
    /*generate random number*/
    int seed = 0;
    printf("Enter the random seed:\n");
    scanf("%d", &seed);
    srand(seed);

    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < MAXN; col++) {
        for (row = 0; row < MAXN; row++) {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }
    printf("Initializing finished...\n");
}


int main(int argc, char** argv){

    int rank;                   /* My process rank           */
    int p;                      /* The number of processes   */
    int generalChunkSize = 0;   /*Define the size of sub matrix*/
    int lastChunkSize = 0;      /*Define the last sub matrix's size*/
    int chunkSize = 0;          /*combine the above two sub chunkSize*/
    int i, j, k, l, m;          /*general variables*/
    int row, col;               /*row number and column number for the matrix*/
    MPI_Status  status;
    double start, end;          /*used to calculate running time*/
    double time[3];

    /*Initialize MPI*/
    MPI_Init(&argc, &argv);
    /*set the rank and processor number for MPI*/
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    /*broadcast the number of rows and cols*/
    //MPI_Bcast(&MAXN, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*calculate the chunk size for each processor*/
    if(MAXN%p == 0)
        generalChunkSize = MAXN/p;
    else
        generalChunkSize = MAXN/p + 1;
    /*the last chunk's size*/
    lastChunkSize = MAXN - (p - 1)*generalChunkSize;
    /*combine the above two different chunk size*/
    chunkSize = (rank == p - 1)? lastChunkSize : generalChunkSize;

    /*local A and B for different processor*/
    float local_A[chunkSize][MAXN], local_B[chunkSize];
    /*buffer Matrices and vectors */
    float buffer_A[MAXN][MAXN] = {0.0}, buffer_B[MAXN] = {0.0};

    if(rank == 0){
        printf("Gaussian Elimination using MPI\nMatrix dimension = %d\n", MAXN);

        /*time start*/
        start = MPI_Wtime();

        /*In processor 0, initialize all the data*/
        initialize_inputs();

        /*time for initializing Matrix*/
        time[0] = MPI_Wtime();
        printf("Initialize the Matrix and Vector takes %f s.\n", time[0] - start);

        /*set local A and local B for processor 0*/
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < MAXN; j++)
                local_A[i][j] = A[i][j];

        for(i = 0; i < chunkSize; i++)
            local_B[i] = B[i];

        /*distribute all the data to other processors*/
        for(i = chunkSize; i < MAXN; i++){
            MPI_Send(&A[i], MAXN, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
            MPI_Send(&B[i], 1, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
        }
    }
    else{
        /*other processors receive the data from processor 0, and store it in local_A and local_b*/
        for(i = 0; i < chunkSize; i++){
            MPI_Recv(&local_A[i], MAXN, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
            MPI_Recv(&local_B[i], 1, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /*The running time of distribuing data*/
    if(rank == 0){
        time[1] = MPI_Wtime();
        printf("Distributing data to each processor takes %f s.\n", time[1] - time[0]);
    }
    /*Gaussian elimination without pivoting*/
    for(i = 0; i < chunkSize; i++){
        /*the original row and col in Matrix A*/
        col = generalChunkSize*rank + i;
        row = col;
        /*for different rank, calculate its multiplier*/
        for(k = col; k < MAXN; k++)
            buffer_A[row][k] = local_A[i][k]/local_A[i][col];
        buffer_B[row] = local_B[i]/local_A[i][col];
        /*boardcast multiplier*/
        MPI_Bcast(&buffer_A[row], MAXN , MPI_FLOAT, rank, MPI_COMM_WORLD);
        MPI_Bcast(&buffer_B[row], 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
    }

    /*for each chunk, doing gaussian elimination*/
    for(i = 0; i < chunkSize; i++){
        row = rank*generalChunkSize + i;
        /*j represents the row number of buffer,
        for rank n, it just need to count the buffer smaller than
        its row number*/
        for(j = 0; j < row; j++){
            for(k = j; k < MAXN; k++){
                local_A[i][k] -= local_A[i][j]*buffer_A[j][k];
            }
            local_B[i] -= local_B[i]*buffer_B[j];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /*running time of gauss elimation*/
    if(rank == 0){
        time[2] = MPI_Wtime();
        printf("Gaussian elimination takes %f s.\n", time[2] - time[1]);
    }

    /*back substitution*/
    for(i = chunkSize - 1; i >= 0; i--){
        /*the original row in the Matrix A, B and X*/
        row = generalChunkSize*rank + i;
        /*get x initialized*/
        X[row]=local_B[i];
        /*broadcast matrix X to all the processors*/
        MPI_Bcast(&X[row], 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
    }

    /*for each row, doing back substitution*/
    for(i = chunkSize - 1; i >= 0; i--){
        row = generalChunkSize*rank + i;
        for(k = MAXN - 1; k > row; k--)
            X[row] -= local_A[i][k]*X[k];
        X[row] /= local_A[i][i];
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        /*time end*/
    	end = MPI_Wtime();
    	/*running time of back substitution*/
    	printf("Back substitution takes %f s.\n", end - time[2]);
    	/*total runing time*/
        printf("Total Running time of this program is %f s.\n", end - start);
    }


    MPI_Finalize();

    return 0;
}
