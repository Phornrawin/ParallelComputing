
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <math.h>
#include <stdio.h>

__global__ void sum_matrix(int* d_x, int* d_y, int* d_z, int n) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	d_z[x + y] = d_x[x + y] + d_y[x + y];	
}

int main()
{
	const int N = 10;
	int* h_x = new int[N*N];
	int* h_y = new int[N*N];
	int* h_z = new int[N*N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_x[i] = i;
			h_y[i] = i;
			h_z[i] = 0;
		}
	}

	int* d_x;
	int* d_y;
	int* d_z;
	cudaMalloc((void **)&d_x, sizeof(int)*(N*N));
	cudaMalloc((void **)&d_y, sizeof(int)*(N*N));
	cudaMalloc((void **)&d_z, sizeof(int)*(N*N));
	cudaMemcpy(d_x, &h_x, sizeof(int)*(N*N), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, &h_y, sizeof(int)*(N*N), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, &h_z, sizeof(int)*(N*N), cudaMemcpyHostToDevice);

	//define DUDA Timer
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start Cuda timer
	cudaEventRecord(start, 0);


	//stencil<<<1, 1 >>> (d_in, d_out, N, k);  
	//fast_stencil <<<blockNum, blockSize>>> (d_in, d_out, N, k);
	dim3 dimGrid(1, 1, 1);
	dim3 dimBlock(10, 10, 1);
	sum_matrix<< <dimGrid, dimBlock>> > (d_x, d_y, d_z, N);


	//stop Cuda timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//compute elapsed time
	float time;
	cudaEventElapsedTime(&time, start, stop);


	cudaMemcpy(&h_x, d_x, N*N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_y, d_x, N*N * sizeof(int), cudaMemcpyDeviceToHost);
	

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
		}
		printf("Out[%d, %d] = %d \n", i, j, h_z[i]);	
	}

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	//report time in kernel
	printf("Time in kernel = %f ms \n", time);



	delete[] h_x;
	delete[] h_y;
	delete[] h_z;

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.


