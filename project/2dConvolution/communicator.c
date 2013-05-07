#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#include "fft.h"
#include "wr_file.h"

#define N 512


int main(int argc, char **argv)
{
    int i, j, k;
    double start, end;
    /*time*/
    double time[9];
    int chunkSize;
    MPI_Status status;
    /*used in fft*/
    float data[N][N];
    /*used in mm*/
    float data_1[N][N], data_2[N][N];
    /*local matrix for fft*/
    float local_data[N][N];

    /*world rank and processor, related to MPI_COMM_WORLD*/
    int world_rank;
    int world_processor;

    /*divided rank and processors for communication, related to taskcomm*/
    int task_rank;
    int task_processor;

    /*This complex array is used to store the temp row to operate FFT*/
    complex temp_data[N];

    /*Initialize rank and number of processor for the MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_processor);

    /*Initialize a new vector for distribute column*/
    MPI_Datatype column, col;
    /*column vector*/
    MPI_Type_vector(N, 1, N, MPI_FLOAT, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, 1*sizeof(float), &column);
    MPI_Type_commit(&column);

    int task = world_rank%4;
    MPI_Comm taskcomm;
    /*split the MPI_COMM_WORLD*/
    MPI_Comm_split(MPI_COMM_WORLD, task, world_rank, &taskcomm);
    MPI_Comm_rank(taskcomm, &task_rank);
    MPI_Comm_size(taskcomm, &task_processor);

    /*inter communicator*/
    MPI_Comm t1_t3_comm, t2_t3_comm, t3_t4_comm;

    /*get chunkSize, in this case chunkSize = 256*/
    chunkSize = N/task_processor;

    /*get the start time of all program*/
    if(world_rank == 0){
        printf("2D convolution using MPI task parallel\n");
        start = MPI_Wtime();
    }

    /*each group do its work and send result by inter communicator*/
    if(task == 0){
        //task 1
        /*create inter communicator for task 1 and task 3*/
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 1, &t1_t3_comm);

        if(task_rank == 0){
            time[0] = MPI_Wtime();

            /*read file*/
            readIm1File(data);
            time[1] = MPI_Wtime();

            printf("Group 1: Reading file 1_im1 takes %f s.\n", time[1] - time[0]);
        }

        /*scatter data to local ranks*/
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        /*Compute time for distributing data*/
        if(task_rank == 0){
            time[2] = MPI_Wtime();
            printf("Group 1: Scatter 1_im1(row) to each processor takes %f s.\n", time[2] - time[1]);
        }

        /*do 1_im1 2dfft*/
        /*Row FFT*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each row for im1*/
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        /*gather all the data and distribute in column*/
        if(task_rank == 0){
            time[3] = MPI_Wtime();
            printf("Group 1: FFT each row for 1_im1 takes %f s.\n", time[3] - time[2]);
        }

        /*gather all the data of 1_im1*/
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_rank == 0){
            time[4] = MPI_Wtime();
            printf("Group 1: Gather all the data of 1_im1(row) takes %f s.\n", time[4] - time[3]);
        }

        /*scatter all the data to column local data*/
        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_rank == 0){
            time[5] = MPI_Wtime();
            printf("Group 1: Scatter 1_im1(column) to each processor takes %f s.\n", time[5] - time[4]);
        }

        /*column FFT*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each column for im1*/
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        /*Gather all the columns from each rank*/
        if(task_rank == 0){
            time[6] = MPI_Wtime();
            printf("Group 1: FFT each column for 1_im1 takes %f s.\n", time[6] - time[5]);
        }

        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        /*Compute time and distribute data to do matrix multiply*/
        if(task_rank == 0){
            time[7] = MPI_Wtime();
            printf("Group 1: Gather all the data of 1_im1(column) takes %f s.\n", time[7] - time[6]);
            /*total time*/
            printf("Group 1: Total time for task 1 in group 1 takes %f s.\n", time[7] - time[0]);

            /*send data to group 3 via inter communicator*/
            MPI_Send(data, N*N, MPI_FLOAT, task_rank, 13, t1_t3_comm);
        }
    }
    else if(task == 1){
        //task 2
        /*create inter communicator for task 2 and task 3*/
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 2, &t2_t3_comm);

        if(task_rank == 0){
            time[0] = MPI_Wtime();

            /*read file*/
            readIm2File(data);
            time[1] = MPI_Wtime();

            printf("Group 2: Reading file 1_im2 takes %f s.\n", time[1] - time[0]);
        }

        /*scatter data to local ranks*/
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        /*Compute time for distributing data*/
        if(task_rank == 0){
            time[2] = MPI_Wtime();
            printf("Group 2: Scatter 1_im2(row) to each processor takes %f s.\n", time[2] - time[1]);
        }

        /*do 1_im1 2dfft*/
        /*Row FFT*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each row for im1*/
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        /*gather all the data and distribute in column*/
        if(task_rank == 0){
            time[3] = MPI_Wtime();
            printf("Group 2: FFT each row for 1_im2 takes %f s.\n", time[3] - time[2]);
        }

        /*gather all the data of 1_im1*/
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_rank == 0){
            time[4] = MPI_Wtime();
            printf("Group 2: Gather all the data of 1_im2(row) takes %f s.\n", time[4] - time[3]);
        }

        /*scatter all the data to column local data*/
        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_rank == 0){
            time[5] = MPI_Wtime();
            printf("Group 2: Scatter 1_im2(column) to each processor takes %f s.\n", time[5] - time[4]);
        }

        /*column FFT*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each column for im1*/
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        /*Gather all the columns from each rank*/
        if(task_rank == 0){
            time[6] = MPI_Wtime();
            printf("Group 2: FFT each column for 1_im2 takes %f s.\n", time[6] - time[5]);
        }

        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        /*Compute time and distribute data to do matrix multiply*/
        if(task_rank == 0){
            time[7] = MPI_Wtime();
            printf("Group 2: Gather all the data of 1_im2(column) takes %f s.\n", time[7] - time[6]);
            /*total time*/
            printf("Group 2: Total time for task 2 in group 2 takes %f s.\n", time[7] - time[0]);
            /*send data to group 3 via inter communicator*/
            MPI_Send(data, N*N, MPI_FLOAT, task_rank, 23, t2_t3_comm);
        }
    }
    else if(task == 2){
        //task 3
        /*local matrix for matrix multiplication*/
        float local_data2[chunkSize][N];
        /*create inter communicator for task 1 and task3, task 2 and task 3, task 3 and task 4*/
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 0, 1, &t1_t3_comm);
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 1, 2, &t2_t3_comm);
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 3, 3, &t3_t4_comm);

        /*receive data from group 1 and group 2*/
        if(task_rank == 0){
            time[0] = MPI_Wtime();

            MPI_Recv(data_1, N*N, MPI_FLOAT, task_rank, 13, t1_t3_comm, &status);
            MPI_Recv(data_2, N*N, MPI_FLOAT, task_rank, 23, t2_t3_comm, &status);

            time[1] = MPI_Wtime();

            /*time of receiving data from group 1 and group 2*/
            printf("Group 3: Receive data from group 1 and group 2 takes %f s.\n", time[1] - time[0]);
        }

        /*do matrix multiplication*/
        MPI_Scatter(data_1, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);
        /*Broadcast data2 to all the ranks*/
        MPI_Bcast(data_2, N*N, MPI_FLOAT, 0, taskcomm);

        if(task_rank == 0){
            time[2] = MPI_Wtime();
            printf("Group 3: Scatter data for multiplication takes %f s.\n", time[2] - time[1]);
        }

        /*Matrix multiplication*/
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < N; j++)
                for(k = 0; k < N; k++)
                    local_data2[i][j] += local_data[i][k]*data_2[k][j];

        /*collect multiplication result from each rank*/
        if(task_rank == 0){
            time[3] = MPI_Wtime();
            printf("Group 3: Matrix multiplication in takes %f s.\n", time[3] - time[2]);
        }

        /*gather data*/
        MPI_Gather(local_data2, chunkSize*N, MPI_FLOAT,
                   data, chunkSize*N, MPI_FLOAT,
                   0, taskcomm);

        if(task_rank == 0){
            time[4] = MPI_Wtime();
            printf("Group 3: Gather data after Matrix multiplication takes %f s.\n", time[4] - time[3]);
            /*total time*/
            printf("Group 3: Total time for task 3 in group 3 takes %f s.\n", time[4] - time[0]);
            /*send result of matrix multiplication to group 4*/
            MPI_Send(data, N*N, MPI_FLOAT, task_rank, 34, t3_t4_comm);
        }

        MPI_Comm_free(&t1_t3_comm);
        MPI_Comm_free(&t2_t3_comm);
    }
    else{
        //task 4
        /*create inter communicator for task 3 and task 4*/
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 3, &t3_t4_comm);

        /*receive data from group 3*/
        if(task_rank == 0){
            time[0] = MPI_Wtime();

            MPI_Recv(data, N*N, MPI_FLOAT, task_rank, 34, t3_t4_comm, &status);

            time[1] = MPI_Wtime();
            printf("Group 4: Receive data from group 3 takes %f s.\n", time[1] - time[0]);
        }

        /*scatter data to each processor*/
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_rank == 0){
            time[2] = MPI_Wtime();
            printf("Group 4: Scatter data(row) to each processor takes %f s.\n", time[2] - time[1]);
        }

        /*Inverse-2DFFT(row)*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each row for im1*/
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, 1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        if(task_rank == 0){
            time[3] = MPI_Wtime();
            printf("Group 4: Inverse-2DFFT(row) takes %f s.\n", time[3] - time[2]);
        }
        /*gather all the data*/
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_rank == 0){
            time[4] = MPI_Wtime();
            printf("Group 4: Gather data of Inverse-2DFFT(row) takes %f s.\n", time[4] - time[3]);
        }

        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_rank == 0){
            time[5] = MPI_Wtime();
            printf("Group 4: Scatter data(column) to each processor takes %f s.\n", time[5] - time[4]);
        }

        /*Inverse-2DFFT(column) for output file*/
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /*fft each column for im1*/
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, 1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        if(task_rank == 0){
            time[6] = MPI_Wtime();
            printf("Group 4: Inverse-2DFFT(column) takes %f s.\n", time[6] - time[5]);
        }

        /*Gather all the columns of output file from each rank*/
        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        if(task_rank == 0){
            time[7] = MPI_Wtime();
                printf("Group 4: Gather data of Inverse-2DFFT(column) takes %f s.\n", time[7] - time[6]);

            writeFile(data);
            time[8] = MPI_Wtime();
            printf("Group 4: Write file to out_1 takes %f s.\n", time[8] - time[7]);
        }
        MPI_Comm_free(&t3_t4_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(world_rank == 0){
        end = MPI_Wtime();
        printf("Total running time of 2D convolution using MPI task parallel takes %f s.\n", end - start);
    }

    /*free vector and task comm*/
    MPI_Type_free(&column);
    MPI_Type_free(&col);
    MPI_Comm_free(&taskcomm);
    MPI_Finalize();
    return 0;
}
