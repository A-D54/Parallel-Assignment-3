#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>


double* randomizeMat(int r, int c) {
    double *mat = (double *)malloc(sizeof(double)*r*c);
    for (int i = 0; i < r * c; i++) {
        mat[i] = (double)rand() / RAND_MAX*100;
    }
    return mat;
}


void matMul(int x, int n, int y, double* A, double* B, double* C) {
    #pragma acc data copyin(A[0:x*n], B[0:n*y]) copyout(C[0:x*y])
    {
        #pragma acc parallel
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                double total = 0;
                for (int k = 0; k < n; k++) {
                    total += A[i*n+k] * B[k*y+j];
                }
                C[i*y+j] = total;
            }
        }
    }
}

int main() {

 int x = 900, n = 500, y = 700;

    clock_t start_time, end_time;
    double total_time = 0;

    double* A = randomizeMat(x, n);
    double* B = randomizeMat(n, y);
    double* Z = (double*)malloc(sizeof(double)*x*y);




    start_time = clock();
    matMul(x, n, y, A, B, Z);
    end_time = clock();
   
    total_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    free(A);
    free(B);
    free(Z);    




    printf("Time Taken: %.4f seconds\n", total_time);
    return 0;
   
   
   
   
   
   
}
