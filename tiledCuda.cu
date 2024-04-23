#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16

double* randomizeMat(int r, int c) {
    double *mat = (double *)malloc(r*c*sizeof(double));
    for (int i = 0; i < r * c; i++) {
        mat[i] = (double)rand()/RAND_MAX*100;
    }
    return mat;
}

__global__ void matmul(double *A, double *B, double *Z, int x, int n, int y) {
    __shared__ double Atile[16][16];
    __shared__ double Btile[16][16];

    int r = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    double total = 0.0;

    for (int t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int idxA = r * n + t * TILE_WIDTH + threadIdx.x;
        if (idxA / n == r && idxA % n < n) {  
            Atile[threadIdx.y][threadIdx.x] = A[idxA];
        } else {
            Atile[threadIdx.y][threadIdx.x] = 0.0;
        }
        int idxB = (t * TILE_WIDTH + threadIdx.y) * y + c;
        if (idxB/y==t * TILE_WIDTH + threadIdx.y && idxB%y==c) {
            Btile[threadIdx.y][threadIdx.x] = B[idxB];
        } else {
            Btile[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();  
        for (int k = 0; k < TILE_WIDTH; k++) {
            total += Atile[threadIdx.y][k] * Btile[k][threadIdx.x];
        }
        __syncthreads();  
    }

    if (r < x && c < y) {
        Z[r * y + c] = total;
    }
}

int main() {
    int x = 500, n = 300, y = 400;
    int Adim = sizeof(double)*x*n;
    int Bdim = sizeof(double)*n*y;
    int Cdim = sizeof(double)*x*y;
    double* A = randomizeMat(x, n);
    double* B = randomizeMat(n, y);
    double* C = (double *)malloc(Cdim);



    double *AA, *BB, *CC;
    cudaMalloc(&AA, Adim);
    cudaMalloc(&BB, Bdim);
    cudaMalloc(&CC, Cdim);
    cudaMemcpy(AA, A, Adim, cudaMemcpyHostToDevice); //COPY BOTH A AND B TO DEVICE
    cudaMemcpy(BB, B, Bdim, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((y+TILE_WIDTH-1)/TILE_WIDTH,(x+TILE_WIDTH-1)/TILE_WIDTH);

    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    float ms = 0, totalSeconds = 0;
   
    cudaEventRecord(start_time);
    matmul<<<dimGrid, dimBlock>>>(AA, BB, CC, x, n, y);
    cudaEventRecord(end_time);
    cudaEventElapsedTime(&ms, start_time, end_time);
    totalSeconds = ms/1000.0;

    printf("Time Taken: %.4f seconds\n", totalSeconds);

    cudaMemcpy(C, CC, Cdim, cudaMemcpyDeviceToHost);

    cudaFree(AA);
    cudaFree(BB);
    cudaFree(CC);

    return 0;
}
