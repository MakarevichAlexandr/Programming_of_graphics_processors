#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <stdio.h>
#include <cublas_v2.h>

#define N (1 << 8)
#define CUDA_CHECK_RETURN(value) {\
	cudaError_t _m_cudaStat = value;\
	if(_m_cudaStat != cudaSuccess) {\
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
		exit(1);\
	}\
}

struct saxpy_functor {
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__ float operator()(float x, float y) {
		return a * x + y;
	}
};

void saxpy(float a, thrust::device_vector<float>& x, thrust::device_vector<float>& y) {
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}


__global__ void gSaxpy(float alpha, float *x, float *y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	y[i] = alpha * x[i] + y[i];
}

float saxpyCUDAC() {
	cudaEvent_t start, stop;
	float *x_d, *x_h, *y_h, *y_d;
	float elapsedTime;
	
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	
	CUDA_CHECK_RETURN(cudaMalloc((void**)&x_d, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&y_d, N * sizeof(float)));
	x_h = (float*)calloc(N, sizeof(float));
	y_h = (float*)calloc(N, sizeof(float));

	for(int i = 0; i < N; i++) {
		x_h[i] = i;
		y_h[i] = 0.87;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(x_d, x_h, N * sizeof(float), cudaMemcpyHostToDevice));	
	CUDA_CHECK_RETURN(cudaMemcpy(y_d, y_h, N * sizeof(float), cudaMemcpyHostToDevice));	

	cudaEventRecord(start, 0);
	gSaxpy <<< N / 256, 256 >>> (3.0, x_d, y_d);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	cudaEventRecord(stop, 0);
	
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	CUDA_CHECK_RETURN(cudaMemcpy(y_h, y_d, N * sizeof(float), cudaMemcpyDeviceToHost));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cudaFree(x_d);
	cudaFree(y_d);
	free(x_h);
	free(y_h);
	
	return elapsedTime;
}

float saxpyCuBLAS() {
	cudaEvent_t start, stop;
	float *x_h, *y_h, *x_d, *y_d;
	float elapsedTime;
	
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));
	
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&x_h, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&y_h, N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&x_d,  N * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&y_d,  N * sizeof(float)));
	
	for(int i = 0; i < N; i++) {
		x_h[i] = (float) i;
		y_h[i] = 0.87f;
	}
	
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);
	
	const int num_rows = N;
	const int num_cols = 1;
	const size_t elem_size = sizeof(float);
	
	cublasSetMatrix(num_rows, num_cols, elem_size, x_h, num_rows, x_d, num_rows);
	cublasSetMatrix(num_rows, num_cols, elem_size, y_h, num_rows, y_d, num_rows);
	
	const int stride = 1;
	float alpha = 3.0f;
	
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	cublasSaxpy(cublas_handle, N, &alpha, x_d, stride, y_d, stride);
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	
	cublasGetMatrix(num_rows, num_cols, elem_size, x_d, num_rows, x_h, num_rows);
	cublasGetMatrix(num_rows, num_cols, elem_size, y_d, num_rows, y_h, num_rows);
	
	//print_array(x_h, y_h, (1 << 8), "Intermediate Set");
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cublasDestroy(cublas_handle);
	CUDA_CHECK_RETURN(cudaFreeHost(x_h));  
	CUDA_CHECK_RETURN(cudaFreeHost(y_h)); 
	CUDA_CHECK_RETURN(cudaFree(x_d));  
	CUDA_CHECK_RETURN(cudaFree(y_d)); 
	
	return elapsedTime;
}

float saxpyTHRUST() {
	cudaEvent_t start, stop;
	float elapsedTime;
	
	CUDA_CHECK_RETURN(cudaEventCreate(&start));
	CUDA_CHECK_RETURN(cudaEventCreate(&stop));

	thrust::host_vector<float> h1(N);
	thrust::host_vector<float> h2(N);
	thrust::sequence(h1.begin(), h1.end());
	thrust::fill(h2.begin(), h2.end(), 0.87);
	
	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;
	
	CUDA_CHECK_RETURN(cudaEventRecord(start, 0));
	saxpy(3.0, d1, d2);
	CUDA_CHECK_RETURN(cudaEventRecord(stop, 0));
	
	CUDA_CHECK_RETURN(cudaEventSynchronize(stop));
	CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, start, stop));
	
	h2 = d2;
	h1 = d1;

	//for(int i = 0; i < (1 << 5); i++)
	//	printf("%d\t%g\t%g\n", i, h1[i], h2[i]);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return elapsedTime;
}

int main() {
	float elapsedTime;
	elapsedTime = saxpyCUDAC();
	printf("CUDA C Time: %f ms\n", elapsedTime);
	
	elapsedTime = saxpyTHRUST();
	printf("THRUST Time: %f ms\n", elapsedTime);
	
	elapsedTime = saxpyCuBLAS();
	printf("cuBLAS Time: %f ms\n", elapsedTime);
		
	return 0;
}
