#include <cstdio>
#include <cuda.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#define SIZE 10
#define V 0.1
#define T 1

using namespace std;

__global__ void kernel(float *f, float *res) {
	int cur = threadIdx.x + blockDim.x * blockIdx.x;
	int prev = cur - 1;
	if (prev == -1)
		prev = SIZE - 1;
	if (cur >= 0 && cur < SIZE) {
		res[cur] = f[cur] + (V * T) * (f[prev] - f[cur]);
	}
}

struct saxpy_functor {
	const float a;
	saxpy_functor(float _a) : a(_a) {}
	__host__ __device__ float operator()(float x, float y) { return a * x + y; }
};

void saxpy(float a, thrust::device_vector<float> &x,
	thrust::device_vector<float> &y) {
	saxpy_functor func(a);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);
}

int main() {
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float F[SIZE];
	float *frez;
	float *tempa;
	float DATA[SIZE];
	for (int i = 0; i < SIZE; i++) {
		DATA[i] = rand() % 10;
		F[i] = DATA[i];
	}

	cudaMalloc((void **)&frez, sizeof(float) * SIZE);
	cudaMalloc((void **)&tempa, sizeof(float) * SIZE);

	cudaMemcpy(tempa, F, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		kernel<<<1, SIZE>>>(tempa, frez);
		cudaMemcpy(F, frez, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
		cudaMemcpy(tempa, frez, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "Time (Raw CUDA C) %g\n", elapsedTime);

	thrust::host_vector<float> h1(SIZE);
	thrust::host_vector<float> h2(SIZE);

	for (int i = 0; i < SIZE; i++) {
		h1[i] = DATA[i];
		if ((i - 1) >= 0)
			h2[i] = DATA[i - 1];
		else
			h2[i] = DATA[SIZE - 1];
		h2[i] = h2[i] * V * T;
	}
	thrust::device_vector<float> d1 = h1;
	thrust::device_vector<float> d2 = h2;

	cudaEventRecord(start, 0);

	for (int j = 0; j < 100; j++) {
		saxpy(1 - V * T, d1, d2);
		h2 = d2;
		d1 = h2;
		for (int i = 0; i < SIZE; i++) {
			if ((i - 1) >= 0)
				h1[i] = h2[i - 1];
			else
				h1[i] = h2[SIZE - 1];
			h1[i] = h1[i] * V * T;
		}
		d2 = h1;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	fprintf(stderr, "Time (Trust SAXPY) %g\n", elapsedTime);
}
