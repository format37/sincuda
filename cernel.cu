
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include "cuda.h"
#include "../../common/book.h"

__global__ void kernel(float *resless, float *res)
{
	//float res[90*314];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	float real, patt, x;
	x = (float)tid / 100;
	real = (1.2*sin(5.01*x) + 2.3*sin(17 * x)) / (1.2 + 2.3);
	//calculate
	patt = sin((float)bid*x);
	res[bid*314+tid] = (real > patt ? real - patt : patt - real);
	__syncthreads();
	//reduce to self
	int index = 314 / 2;
	while (index != 0)
	{
		if (tid < index) res[bid * 314 + tid] += res[bid * 314 + tid + index];
		__syncthreads();
		index /= 2;
	}
	//reduce less
	if (tid == 0&&bid<90) resless[bid] = res[bid*314];
}

int main(void)
{
	float resless[90],*dev_resless,res[90*314],*dev_res;
	//float sig[314];
	/*cudaEvent_t     start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));*/
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffer
	cudaStatus = cudaMalloc((void**)&dev_resless, 90 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_res, 314*90 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//// Copy input vectors from host memory to GPU buffers.
	//cudaStatus = cudaMemcpy(dev_res, res, 90 * sizeof(float), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	fprintf(stderr, "cudaMemcpy failed!");
	//	goto Error;
	//}
	//sinnatvol++
	kernel << <90,314 >> >(dev_resless,dev_res);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(resless, dev_resless, 90 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(res, dev_res, 314*90 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//for (x = 0; x < 3.14; x = x + 0.01)
	/*for (xj = 0; xj < 314; xj ++)
	{
		x = xj / 100;
		real = (1.2*sin(5.01*x) + 2.3*sin(17 * x)) / (1.2 + 2.3);
		for (i = 1; i < 90; i++)
		{
			patt = sin(i*x);
			res[i] = res[i] + (real > patt ? real - patt : patt - real);
		}
	}*/

	//sinnatvol--
	// get stop time, and display the timing results
	/*HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));*/

	// free memory on the gpu side
	cudaStatus = cudaFree(dev_resless);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed!");
		goto Error;
	}
	cudaStatus = cudaFree(dev_res);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaFree failed!");
		goto Error;
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	// free memory on the cpu side
	free(resless);
	//free(res);

    return 0;

Error:
	cudaFree(dev_resless);

	return cudaStatus;
}
