#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16

double* randomizeMat(int s) {
    double* mat = (double*)malloc(s*sizeof(double));
    for (int i = 0; i < s; i++) {
        mat[i] = (double)rand() / RAND_MAX * 100;
    }
    return mat;
}


__global__ void matMul(double *A, double *B, double *Z, int x, int n, int y) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < x && c < y) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[r * n + k] * B[k*y+ c];
        }
        Z[r * y + c] = sum;
    }
}

int main() {
    int x = 500, n = 300, p = 400;
   
    int Adim = sizeof(double)*x*n;
    int Bdim = sizeof(double)*n*y;
    int Zdim = sizeof(double)*x*y;
    double* A = randomizeMat(x, n);
    double* B = randomizeMat(n, y);
    double* Z = (double *)malloc(Zdim);

    double *AA, *BB, *ZZ;
    cudaMalloc(&AA, Adim);
    cudaMalloc(&BB, Bdim);
    cudaMalloc(&ZZ, Zdim);
    cudaMemcpy(AA, A, Adim, cudaMemcpyHostToDevice);
    cudaMemcpy(BB, B, Bdim, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((y + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (x + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    float ms = 0, total_time = 0;
    cudaEventRecord(start_time);
    matMul<<<numBlocks, threadsPerBlock>>>(AA, BB, ZZ, x, n, y);
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);

    cudaEventElapsedTime(&ms, start_time, end_time);
    total_time += ms / 1000.0;

    printf("Time Taken: %.4f seconds\n", total_time);

    cudaMemcpy(Z, ZZ, Zdim, cudaMemcpyDeviceToHost);

    cudaFree(AA);
    cudaFree(BB);
    cudaFree(ZZ);

    return 0;
}
