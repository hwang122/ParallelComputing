#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "mpi.h"
#include "omp.h"

#include "fft.h"

#define N 512


void readFile(float (*a)[N], float (*b)[N]){
    int i, j;
    FILE *f1, *f2;

    f1 = fopen("1_im1", "r");
    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++){
            fscanf(f1, "%g", a[i][j]);
            fscanf(f2, "%g", b[i][j]);
        }

    fclose(f1);
    fclose(f2);
}

void writeFile(float (*c)[N]){
    int i, j;
    FILE *f3;

    f3 = fopen("out_1", "w");

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++)
            fprintf(f3, "%.7e", c[i][j]);
        fprintf(f3, "\n");
    }

    fclose(f3);
}

void read_send(int rank, int p){
    int i, j, k;
    int chunkSize;
    MPI_Status status;
    double start, end;
    double time[14];

    /*Set the chunk size for each processor*/
    chunkSize = N/p;

    /*These two arrays are used to store the local data distributed by rank 0*/
    float local_data1[chunkSize][N], local_data2[chunkSize][N];
    /*local matrix for matrix multiplication*/
    float local_data3[chunkSize][N];
    /*This complex array is used to store the temp row to operate FFT*/
    complex temp_data[N];

    /*Initialization of the original Matrix and distribution of data*/
    if(rank == 0){
        /*input data*/
        float data_1[N][N], data_2[N][N];
        /*output data*/
        float output[N][N];

        printf("2D convolution using SPMD model and MPI Send&Receive operations\n");
        start = MPI_Wtime();
        /*Read data from the files*/
        readFile(data_1, data_2);

        time[0] = MPI_Wtime();
        printf("Reading file takes %f s.\n", time[0] - start);
        /*Set data for rank 0*/
        memmove(local_data1, data_1, sizeof(float)*chunkSize*N);
        memmove(local_data2, data_2, sizeof(float)*chunkSize*N);

        /*Distribute data using MPI_Send*/
        for(i = chunkSize; i < N; i++){
            MPI_Send(&data_1[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
            MPI_Send(&data_2[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
        }
    }
    else{
        /*Receive data using MPI_Recv*/
        for(i = 0; i < chunkSize; i++){
            MPI_Recv(&local_data1[i], N, MPI_FLOAT, 0, i%chunkSize, MPI_COMM_WORLD, &status);
            MPI_Send(&local_data2[i], N, MPI_FLOAT, 0, i%chunkSize, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /*Compute time for distributing data*/
    if(rank == 0){
        time[1] = MPI_Wtime();
        printf("Distributing data(row) to each processor takes %f s.\n", time[1] - time[0]);
    }

    /*Row FFT*/
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
    /*Initialize a new vector for distribute column*/
    MPI_Datatype column;
    /*column vector*/
    MPI_Type_vector(N, chunkSize, N, MPI_FLOAT, &column);
    MPI_Type_commit(&column);

    /*gather all the data and distribute in column*/
    if(rank == 0){
        time[2] = MPI_Wtime();
        printf("FFT each row for input im1 and im2 takes %f s.\n", time[2] - time[1]);
        /*gather all the message to rank 0*/
        for(i = chunkSize; i < N; i++){
            MPI_Recv(&data_1[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD, &status);
            MPI_Recv(&data_2[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD, &status);
        }

        time[3] = MPI_Wtime();
        printf("Gather all the data from different ranks(row) takes %f s.\n", time[3] - time[2]);

        /*distribute data in column*/
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < N; j++){
                local_data1[i][j] = data_1[j][i];
                local_data2[i][j] = data_2[j][i];
            }

        /*Using new vector to send data*/
        for(i = 1; i < p; i++){
            MPI_Send(&data_1[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD);
            MPI_Send(&data_2[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD);
        }

    }
    else{
        /*send rows have been FFT to rank 0*/
        for(i = 0; i < chunkSize; i++){
            MPI_Send(&local_data1[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
            MPI_Send(&local_data1[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);
        }

        /*receive data according to new vector column*/
        MPI_Recv(&local_data1[0], 1, column, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&local_data2[0], 1, column, 0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[4] = MPI_Wtime();
        printf("Distributing data(column) to each processor takes %f s.\n", time[4] - time[3]);
    }
    /*column FFT*/
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
    /*Gather all the columns from each rank*/
    if(rank == 0){
        time[5] = MPI_Wtime();
        printf("FFT each column for input im1 and im2 takes %f s.\n", time[5] - time[4]);

        for(i = 1; i < p; i++){
            MPI_Recv(&data_1[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&data_2[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD, &status);
        }
    }
    /*Send columns into rank 0*/
    else{
        MPI_Send(&local_data1[0], 1, column, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&local_data1[0], 1, column, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /*Compute time and distribute data to do matrix multiply*/
    if(rank == 0){
        time[6] = MPI_Wtime();
        printf("Gather all the data from different ranks(column) takes %f s.\n", time[6] - time[5]);

        /*matrix multipication*/
        /*set local data for rank 0*/
        memmove(local_data1, data_1, sizeof(float)*chunkSize*N);

        /*Send data1 by row to each rank*/
        for(i = chunkSize; i < N; i++)
            MPI_Send(&data_1[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD);
        /*Broadcast data2 to all the ranks*/
        MPI_Bcast(&data_2[0], N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else{
        /*receive data1 from rank 0*/
        for(i = 0; i < chunkSize; i++)
            MPI_Recv(&local_data1[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[7] = MPI_Wtime();
        printf("Distributing data for multiplication takes %f s.\n", time[7] - time[6]);
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
            c_fft1d(temp_data, N, 1);
            local_data3[i][j] = temp_data[j].r;
        }
    }

    if(rank == 0){
        time[9] = MPI_Wtime();
        printf("FFT each row for out_1 takes %f s.\n", time[9] - time[8]);

        memmove(output, local_data3, sizeof(float)*chunkSize*N);

        for(i = chunkSize; i < N; i++)
            MPI_Recv(&output[i], N, MPI_FLOAT, i/chunkSize, i%chunkSize, MPI_COMM_WORLD, &status);

        time[10] = MPI_Wtime();
        printf("Gather all the data of FFT output file(row) takes %f s.\n", time[10] - time[9]);

        /*distribute output file in column*/
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < N; j++)
                local_data3[i][j] = output[j][i];

        /*Using new vector to send output file*/
        for(i = 1; i < p; i++)
            MPI_Send(&output[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD);
    }
    else{
        for(i = 0; i < chunkSize; i++)
            MPI_Send(local_data3[i], N, MPI_FLOAT, 0, i, MPI_COMM_WORLD);

        /*receive data according to new vector column*/
        MPI_Recv(&local_data3[0], 1, column, 0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        time[11] = MPI_Wtime();
        printf("Distributing output file(column) to each processor takes %f s.\n", time[11] - time[10]);
    }

    /*Inverse-2DFFT(column) for output file*/
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /*fft each column for im1*/
            temp_data[j].r = local_data3[i][j];
            temp_data[j].i = 0;
            c_fft1d(temp_data, N, 1);
            local_data3[i][j] = temp_data[j].r;
        }
    }

    /*Gather all the columns of output file from each rank*/
    if(rank == 0){
        time[12] = MPI_Wtime();
        printf("FFT each column for output file takes %f s.\n", time[12] - time[11]);

        for(i = 1; i < p; i++)
            MPI_Recv(&output[0][i*chunkSize], 1, column, i, 0, MPI_COMM_WORLD, &status);
    }
    /*Send columns into rank 0*/
    else{
        MPI_Send(&local_data3[0], 1, column, 0, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        time[13] = MPI_Wtime();
        printf("Gather all the data of output file(column) takes %f s.\n", time[13] - time[12]);

        writeFile(output);

        end = MPI_Wtime();
        printf("Write output file to file takes %f s.\n", end - time[13]);

        printf("Total running time of 2D convolution using MPI_Send&MPI_Recv takes %f", end - start);
    }

    /*free vector column*/
    MPI_Type_free(&column);
}

void collective(int rank, int p){
    int i, j, k;
    int chunkSize;
    MPI_Status status;
    double start, end;
    double time[14];

    /*Set the chunk size for each processor*/
    chunkSize = N/p;

    /*These two arrays are used to store the local data distributed by rank 0*/
    float local_data1[chunkSize][N], local_data2[chunkSize][N];
    /*local matrix for matrix multiplication*/
    float local_data3[chunkSize][N];
    /*This complex array is used to store the temp row to operate FFT*/
    complex temp_data[N];

}

int main(int argc, char **argv))
{
    int rank;
    int p;

    /*Initialize rank and number of processor for the MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    read_send(rank, p);

    collective(rank, p);

    MPI_Finalize();
    return 0;
}
