
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void reduceInterleaved(int *d_A, int n) {
	int id = threadIdx.x;
	int s = 1;
	for (int i = 0; i < (int)log2((double)n); i++){
		if (id < n / 2) {
			d_A[(int)pow((double)2, (double)(i + 1)) * id] = d_A[(int)pow((double)2, (double)(i + 1)) * id] + d_A[(int)pow((double)2, (double)(i + 1)) * id + s];
			s = s * 2;
		}
		__syncthreads();
	}
}

__global__ void reduceContiguous(int *d_A, int n) {
	int id = threadIdx.x;
	int s = n/2;
	for (int i = 0; i < (int)log2((double)n); i++) {
		if (id < n / 2) {
			d_A[id] = d_A[id] + d_A[id + s];
			s = s / 2;
		}
		__syncthreads();
	}
}

int main()
{
	const int N = 64;
	int h_A[N];
	for (int i = 0; i < N; i++) {
		h_A[i] = i;
	}
	

	int* d_A;
	cudaMalloc((void **)&d_A, sizeof(int)*(N));
	cudaMemcpy(d_A, &h_A, sizeof(int)*(N), cudaMemcpyHostToDevice);

	//define DUDA Timer
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//start Cuda timer
	cudaEventRecord(start, 0);


	//stencil<<<1, 1 >>> (d_in, d_out, N, k);  
	//fast_stencil <<<blockNum, blockSize>>> (d_in, d_out, N, k);
	reduceContiguous<< <1, 1024 >> > (d_A, N);


	//stop Cuda timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//compute elapsed time
	float time;
	cudaEventElapsedTime(&time, start, stop);


	cudaMemcpy(&h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);


	
	printf("Out[0] = %d \n",h_A[0]);


	cudaFree(d_A);

	//report time in kernel
	printf("Time in kernel = %f ms \n", time);



	return 0;
}

// Helper function for using CUDA to add vectors in parallel.


