#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
/* Matrices and vectors */
float A[MAXN][MAXN], B[MAXN], X[MAXN];
float buffer_A[MAXN][MAXN] = {0}, buffer_B[MAXN] = {0};
/* A * X = B, solve for X */


/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {

    /*generate random number*/
    int seed = 0;
    printf("Enter the random seed:\n");
    scanf("%d", seed);
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
}


main(int argc, char** argv){

    printf("Gaussian Elimination using MPI\n\nMatrix dimension = %d\n", MAXN);

    int rank;           /* My process rank           */
    int p;              /* The number of processes   */
    int chunkSize = 0;  /*Define the size of sub matrix*/
    int i, j, k, l, m;  /*general variables*/
    float temp;         /*used to store temp float value*/
    int row, col        /*row number and column number for the matrix*/
    MPI_Status  status;

    /*Initialize MPI*/
    MPI_Init(&argc, &argv);
    /*set the rank and processor number for MPI*/
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    /*broadcast the number of rows and cols*/
    MPI_Bcast(&MAXN, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*calculate the chunk size for each processor*/
    if(MAXN%p == 0)
        chunkSize = MAXN/p;
    else
        chunkSize = MAXN/p + 1;

    /*local A and B for different processor*/
    float loacl_A[chunkSize][MAXN], local_B[chunkSize];

    if(rank == 0){
        /*In processor 0, initialize all the data*/
        initialize_inputs();
        /*set local A and local B for processor 0*/
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < MAXN; j++)
                local_A[i][j] = A[i][j];

        for(i = 0; i < chunkSize; i++)
            local_B[i] = B[i];

        /*distribute all the data to other processors*/
        for(i = chunkSize; i < MAXN; i++){
            MPI_Send(&A[i][0], MAXN, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
            MPI_Send(B[i], 1, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
        }
    }
    else{
        /*other processors receive the data from processor 0, and store it in local_A and local_b*/
        for(i = 0; i < chunkSize; i++){
            MPI_Recv(&local_A[i][0], MAXN, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &statues);
            MPI_Recv(local_B[i], 1, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &statues);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*Gaussian elimination without pivoting*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < p; j++){
            /*the original row and col in Matrix A*/
            col = chunkSize*rank + i;
            row = col;
            /*for different rank, calculate its multiplier*/
            if(rank = j){
                for(k = col; k < MAXN; k++)
                    buffer_A[row][k] = local_A[i][k]/local_A[i][col];
                buffer_B[row] = local_B[i]/local_A[i][col]
                /*boardcast multiplier*/
                MPI_Bcast(&buffer_A[row][col], MAXN - col, MPI_FLOAT, rank, MPI_COMM_WORLD);
                MPI_Bcast(buffer_B[row], MAXN - col, MPI_FLOAT, rank, MPI_COMM_WORLD);
            }

            /*for each chunk, doing gaussian elimination*/
            for(k = 0; k < chunkSize; k++)
                /*l represents the row number of buffer,
                since only number of processors' buffer is broadcast,
                it can add chunkSize everytime*/
                for(l = i; l < rank*chunkSize + k; l+=chunkSize){
                    for(m = 0; m < MAXN; m++)
                        local_A[k][m] -= local_A[k][col]*buffer_A[l][m];
                local_B[k] -= local_B[k]*buffer_B[l];
                }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*back substitution*/
    for(i = chunkSize - 1; i >= 0; i--)
        for(j = p - 1; j >= 0; j--){
            /*the original row in the Matrix A, B and X*/
			row = i * p + j;
            if(rank == j){
                //get x initialized
                X[row]=local_B[i];
                //broadcast matrix X to all the processors
                MPI_Bcast(X[row], 1, MPI_FLOAT, rank, MPI_COMM_WORLD);
            }
            /*for each row, doing back substitution*/
            for(k = MAXN - 1; k > row; k--){
                X[row] -= local_A[i][k]*X[k];
            }
            X[row] /= local_A[i][i];
		}

    MPI_Finalize();
}
