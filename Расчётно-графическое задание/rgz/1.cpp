#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N (1 << 8)
#define alpha 3.0

int main() {
	float *x_h = (float*)calloc(N, sizeof(float));
	float *y_h = (float*)calloc(N, sizeof(float));

	for(int i = 0; i < N; i++) {
		x_h[i] = i;
		y_h[i] = 0.87;
	}
	
	
	clock_t start = clock();
	for(int i = 0; i < N; i++) {
		y_h[i] = alpha * x_h[i] + y_h[i];
	}
	clock_t stop = clock();

	float elapsedTime = (float)(stop - start);
	printf("OpenGL Time: %f ms\n", elapsedTime);
	
	free(x_h);
	free(y_h);
	
	return 0;
}
