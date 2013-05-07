#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#include "fft.h"
#include "wr_file.h"

#define N 512

void collective(int rank, int p){
    int i, j, k;
    int chunkSize;
    double start, end;
    double time[14];

    /*input data*/
    float data_1[N][N], data_2[N][N];
    /*output data*/
    float output[N][N];
    /*Set the chunk size for each processor*/
    chunkSize = N/p;

    /*These two arrays are used to store the local data distributed by rank 0*/
    float local_data1[N][N], local_data2[N][N];
    /*local matrix for matrix multiplication*/
    float local_data3[chunkSize][N];
    /*This complex array is used to store the temp row to operate FFT*/
    complex temp_data[N];

    /*Initialization of the original Matrix and distribution of data*/
    if(rank == 0){
        printf("2D convolution using SPMD model and MPI Collective operations\n");
        start = MPI_Wtime();
        /*Read data from the files*/
        readFile(data_1, data_2);

        time[0] = MPI_Wtime();
        printf("Reading file takes %f s.\n", time[0] - start);
    }

    /*scatter all the data to local data*/
    MPI_Scatter(data_1, chunkSize*N, MPI_FLOAT,
                local_data1, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(data_2, chunkSize*N, MPI_FLOAT,
                local_data2, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    /*Compute time for distributing data*/
    if(rank == 0){
        time[1] = MPI_Wtime();
        printf("Scatter data(row) to each processor takes %f s.\n", time[1] - time[0]);
    }

    /*Row FFT*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each row for im1*/
            temp_data[j].r = local_data1[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data1[i][j] = temp_data[j].r;

        for(j = 0; j < N; j++){
            /*fft each row for im2*/
            temp_data[j].r = local_data2[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data2[i][j] = temp_data[j].r;
    }

    /*gather all the data and distribute in column*/
    if(rank == 0){
        time[2] = MPI_Wtime();
        printf("FFT each row for input im1 and im2 takes %f s.\n", time[2] - time[1]);
    }

    MPI_Gather(local_data1, chunkSize*N, MPI_FLOAT,
               data_1, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_data2, chunkSize*N, MPI_FLOAT,
               data_2, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if(rank == 0){
        time[3] = MPI_Wtime();
        printf("Gather all the data from different ranks takes %f s.\n", time[3] - time[2]);
    }

    /*Initialize a new vector for distribute column*/
    MPI_Datatype column, col;
    /*column vector*/
    MPI_Type_vector(N, 1, N, MPI_FLOAT, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, 1*sizeof(float), &column);
    MPI_Type_commit(&column);

    /*scatter all the data to column local data*/
    MPI_Scatter(data_1, chunkSize, column,
                local_data1, chunkSize, column,
                0, MPI_COMM_WORLD);
    MPI_Scatter(data_2, chunkSize, column,
                local_data2, chunkSize, column,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[4] = MPI_Wtime();
        printf("Scatter data(column) to each processor takes %f s.\n", time[4] - time[3]);
    }
    /*column FFT*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each column for im1*/
            temp_data[j].r = local_data1[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data1[j][i] = temp_data[j].r;

        for(j = 0; j < N; j++){
            /*fft each column for im2*/
            temp_data[j].r = local_data2[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data2[j][i] = temp_data[j].r;
    }
    /*Gather all the columns from each rank*/
    if(rank == 0){
        time[5] = MPI_Wtime();
        printf("FFT each column for input im1 and im2 takes %f s.\n", time[5] - time[4]);
    }

    MPI_Gather(local_data1, chunkSize, column,
               data_1, chunkSize, column,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_data2, chunkSize, column,
               data_2, chunkSize, column,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    /*Compute time and distribute data to do matrix multiply*/
    if(rank == 0){
        time[6] = MPI_Wtime();
        printf("Gather all the data from different ranks(column) takes %f s.\n", time[6] - time[5]);
    }

    MPI_Scatter(data_1, chunkSize*N, MPI_FLOAT,
                local_data1, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    /*Broadcast data2 to all the ranks*/
    MPI_Bcast(data_2, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[7] = MPI_Wtime();
        printf("Scatter data for multiplication takes %f s.\n", time[7] - time[6]);
    }

    /*Matrix multiplication*/
    for(i = 0; i < chunkSize; i++)
        for(j = 0; j < N; j++)
            for(k = 0; k < N; k++)
                local_data3[i][j] += local_data1[i][k]*data_2[k][j];

    /*collect multiplication result from each rank*/
    if(rank == 0){
        time[8] = MPI_Wtime();
        printf("Matrix multiplication in takes %f s.\n", time[8] - time[7]);
    }

    /*Inverse-2DFFT(row) for output file*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each row for im1*/
            temp_data[j].r = local_data3[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, 1);

        for(j = 0; j < N; j++)
            local_data3[i][j] = temp_data[j].r;
    }

    if(rank == 0){
        time[9] = MPI_Wtime();
        printf("Inverse-2DFFT for out_1(row) takes %f s.\n", time[9] - time[8]);
    }

    MPI_Gather(local_data3, chunkSize*N, MPI_FLOAT,
               output, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        time[10] = MPI_Wtime();
        printf("Gather all the data of Inverse-2DFFT for out_1(row) takes %f s.\n", time[10] - time[9]);
    }

    MPI_Scatter(output, chunkSize, column,
                local_data1, chunkSize, column,
                0, MPI_COMM_WORLD);

    if(rank == 0){
        time[11] = MPI_Wtime();
        printf("Scatter out_1(column) to each processor takes %f s.\n", time[11] - time[10]);
    }

    /*Inverse-2DFFT(column) for output file*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each column for im1*/
            temp_data[j].r = local_data1[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, 1);

        for(j = 0; j < N; j++)
            local_data1[j][i] = temp_data[j].r;
    }

    /*Gather all the columns of output file from each rank*/
    if(rank == 0){
        time[12] = MPI_Wtime();
        printf("Inverse-2DFFT out_1(column) takes %f s.\n", time[12] - time[11]);
    }

    MPI_Gather(local_data1, chunkSize, column,
               output, chunkSize, column,
               0, MPI_COMM_WORLD);

    if(rank == 0){
        time[13] = MPI_Wtime();
        printf("Gather all the data of output file(column) takes %f s.\n", time[13] - time[12]);

        writeFile(output);

        end = MPI_Wtime();
        printf("Write output file to file takes %f s.\n", end - time[13]);

        printf("Total running time of 2D convolution using MPI_Scatter&MPI_Gather takes %f s.\n", end - start);
    }

    /*free vector column*/
    MPI_Type_free(&column);
    MPI_Type_free(&col);
}


int main(int argc, char **argv)
{
    int rank;
    int p;

    /*Initialize rank and number of processor for the MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    collective(rank, p);

    MPI_Finalize();
    return 0;
}