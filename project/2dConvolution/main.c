#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

#include "fft.h"

#define N 512

float data_1[N][N], data_2[N][N];

extern void c_fft1d();

void readFile(){
    int i, j;
    FILE *f1, *f2;

    f1 = fopen("1_im1", "r");
    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++){
            fscanf(f1, "%g", &data_1[i][j]);
            fscanf(f2, "%g", &data_2[i][j]);
        }

    fclose(f1);
    fclose(f2);
}

int main(int argc, char **argv))
{
    int rank;
    int p;
    int i, j;
    int chunkSize;
    MPI_Status status;
    double start, end;
    double time[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    chunkSize = N/p;

    if(rank == 0){
        printf("2D convolution using SPMD model and MPI Send&Receive operations\n");
        start = MPI_Wtime();
        readFile();


    }




    return 0;
}
