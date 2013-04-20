#include <stdio.h>
#include <stdlib.h>

#define N 512

int main()
{
    int i, j;
    FILE *f1, *f2;

    float data_1[N][N], data_2[N][N];

    f1 = fopen("1_im1", "r");
    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++){
            fscanf(f1, "%g", &data_1[i][j]);
            fscanf(f2, "%g", &data_2[i][j]);
        }

    fclose(f1);
    fclose(f2);

    return 0;
}
