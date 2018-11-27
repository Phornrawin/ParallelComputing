
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>


__global__ void hills_scan(int *d_A, int n) {
	int id = threadIdx.x;
	for (int i = 1; i <= (int)log2((double)n); i++) {
		if (id <= n && id >= i)
			d_A[id] = d_A[id - (int)pow((double)2, (double)(i - 1))] + d_A[id];
	}
}

__global__ void parallel_scan(int *d_A, int n) {
	int id = threadIdx.x;
	for (int d = 1; d <= (int)log2((double)n); d++){
		if (id <= (n-1)/(int)pow((double)2, (double)(d + 1)))
			d_A[id + (int)pow((double)2, (double)(d + 1)) - 1] = d_A[id + (int)pow((double)2, (double)(d)) - 1] + d_A[id + (int)pow((double)2, (double)(d)) - 1];
	}
}

int main()
{
	const int N = 8;
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
	parallel_scan << <1, 1024 >> > (d_A, N);


	//stop Cuda timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//compute elapsed time
	float time;
	cudaEventElapsedTime(&time, start, stop);


	cudaMemcpy(&h_A, d_A, N * sizeof(int), cudaMemcpyDeviceToHost);


	for (int i = 0; i < N; i++) {
		printf("Out[%d] = %d \n", i, h_A[i]);
	}
	


	cudaFree(d_A);

	//report time in kernel
	printf("Time in kernel = %f ms \n", time);



	return 0;
    
}


