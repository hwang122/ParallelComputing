#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"

#include "fft.h"

#define N 512


void readFile(float (*a)[N], float (*b)[N]){
    int i, j;
    FILE *f1, *f2;

    f1 = fopen("1_im1", "r");
    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++){
            fscanf(f1, "%g", &a[i][j]);
            fscanf(f2, "%g", &b[i][j]);
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
    double time[10];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);


    chunkSize = N/p;

    float local_data1[chunkSize][N], local_data2[chunkSize][N];
    //complex cdata_1[N][N], cdata_2[N][N];
    complex temp_data[N];

    if(rank == 0){
        float data_1[N][N], data_2[N][N];

        printf("2D convolution using SPMD model and MPI Send&Receive operations\n");
        start = MPI_Wtime();
        readFile(data_1, data_2);

        time[0] = MPI_Wtime();
        printf("Reading file takes %f s.\n", time[0] - start);

        memmove(local_data1, data_1, sizeof(float)*chunkSize*N);
        memmove(local_data2, data_2, sizeof(float)*chunkSize*N);

        for(i = chunkSize; i < N; i++){
            MPI_Send(&data_1[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
            MPI_Send(&data_2[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
        }
    }
    else{
        for(i = 0; i < chunkSize; i++){
            MPI_Recv(&local_data1[i], N, MPI_FLOAT, 0, i%chunkSize, MPI_COMM_WORLD, &status);
            MPI_Send(&local_data2[i], N, MPI_FLOAT, 0, i%chunkSize, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[1] = MPI_Wtime();
        printf("Distributing data(row) to each processor takes %f s.\n", time[1] - time[0]);
    }

    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each row for im1*/
            temp_data[j].r = local_data1[i][j];
            temp_data[j].i = 0;
            c_fft1d(temp_data, N, -1);
            local_data1[i][j] = temp_data[j].r;

            /*fft each row for im2*/
            temp_data[j].r = local_data2[i][j];
            temp_data[j].i = 0;
            c_fft1d(temp_data, N, -1);
            local_data2[i][j] = temp_data[j].r;
        }
    }

    MPI_Datatype column;
    MPI_Type_vector(N, chunkSize, N, MPI_FLOAT, &column);
    MPI_Type_commit(&column);

    /*gather all the data and distribute in column*/
    if(rank == 0){
        time[2] = MPI_Wtime();
        printf("FFT each row for input im1 and im2 takes %f s.\n", time[2] - time[1]);

        for(i = chunkSize; i < N; i++){
            MPI_Recv(&data_1[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD, &status);
            MPI_Recv(&data_2[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD, &status);
        }

        time[3] = MPI_Wtime();
        printf("Gather all the data from different ranks(row) takes %f s.\n", time[3] - time[2]);

        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < N; j++){
                local_data1[i][j] = data_1[j][i];
                local_data2[i][j] = data_2[j][i];
            }

        for(i = 1; i < p; i++){
            MPI_Send(&data_1[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD);
            MPI_Send(&data_2[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD);
        }

    }
    else{
        for(i = 0; i < chunkSize; i++){
            MPI_Send(&local_data1[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
            MPI_Send(&local_data1[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }

        MPI_Recv(&local_data1[0][0], 1, column, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&local_data2[0][0], 1, column, 0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[4] = MPI_Wtime();
        printf("Distributing data(column) to each processor takes %f s.\n", time[4] - time[3]);
    }

    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each column for im1*/
            temp_data[j].r = local_data1[i][j];
            temp_data[j].i = 0;
            c_fft1d(temp_data, N, -1);
            local_data1[i][j] = temp_data[j].r;

            /*fft each column for im2*/
            temp_data[j].r = local_data2[i][j];
            temp_data[j].i = 0;
            c_fft1d(temp_data, N, -1);
            local_data2[i][j] = temp_data[j].r;
        }
    }

    if(rank == 0){
        time[5] = MPI_Wtime();
        printf("FFT each column for input im1 and im2 takes %f s.\n", time[5] - time[5]);

        for(i = 1; i < p; i++){
            MPI_Recv(&data_1[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&data_2[0][i*chunkSize], 1, columb, i, 0, MPI_COMM_WORLD, &status);
        }
    }
    else{
        MPI_Send(&local_data1[0][0], 1, column, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&local_data1[0][0], 1, column, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[6] = MPI_Wtime();
        printf("Gather all the data from different ranks(column) takes %f s.\n", time[6] - time[5]);

        MPI_Send();
    }
    else{
        MPI_Recv();
    }


    return 0;
}
