#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <getopt.h>

#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF*COEF*2//-(COEF-1)*2
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE/2
#define IMIN(A,B) (A<B?A:B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID IMIN(32,(VERTCOUNT+THREADSPERBLOCK-1)/THREADSPERBLOCK)

typedef float(*ptr_f)(float, float, float);

struct Vertex
{
  float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];
texture<float, 3, cudaReadModeElementType> df_tex;
cudaArray* df_Array = 0;

__device__ float interpolation_pl(float *arr_f, int x_size, int y_size, int z_size, float x, float y, float z)
{
  int interp = 1;
  int vx, vy, vz;
  int vx0, vy0, vz0, vx1, vy1, vz1;
  vx0 = x;
  vx1 = vx0 + 1;
  float x0 = ((float)((int)x)) - FGSHIFT;
  float x1 = x0 + 1.f;
  if (vx1 >= FGSIZE) {
    interp = 0;
    vx = FGSIZE - 1;
  } else {
    vx = ((vx1 < FGSIZE) && ((x - vx0) > 0.5)) ? vx1 : vx0;
  }

  vy0 = y;
  vy1 = vy0 + 1;
  float y0 = ((float)((int)y)) - FGSHIFT;
  float y1 = y0 + 1.f;
  if (vy1 >= FGSIZE) {
    interp = 0;
    vy = FGSIZE - 1;
  } else {
    vy = ((vy1 < FGSIZE) && ((y - vy0) > 0.5)) ? vy1 : vy0;
  }

  vz0 = z;
  vz1 = vz0 + 1;
  float z0 = ((float)((int)z)) - FGSHIFT;
  float z1 = z0 + 1.f;
  if (vz1 >= FGSIZE) {
    interp = 0;
    vz = FGSIZE - 1;
  } else {
    vz = ((vz1 < FGSIZE) && ((z - vz0) > 0.5)) ? vz1 : vz0;
  }
  
  if (interp == 0) {
    return arr_f[z_size * (vx * y_size + vy) + vz];
  } else {
    x -= FGSHIFT;
    y -= FGSHIFT;
    z -= FGSHIFT;
    return arr_f[z_size * (vx0 * y_size + vy0) + vz0]*(x-x1)*(y-y1)*(z-z1)/(x0-x1)/(y0-y1)/(z0-z1) + arr_f[z_size * (vx1 * y_size + vy0) + vz0]*(x-x0)*(y-y1)*(z-z1)/(x1-x0)/(y0-y1)/(z0-z1) + arr_f[z_size * (vx1 * y_size + vy0) + vz1]*(x-x0)*(y-y1)*(z-z0)/(x1-x0)/(y0-y1)/(z1-z0) + arr_f[z_size * (vx1 * y_size + vy1) + vz0]*(x-x0)*(y-y0)*(z-z1)/(x1-x0)/(y1-y0)/(z0-z1) + arr_f[z_size * (vx1 * y_size + vy1) + vz1]*(x-x0)*(y-y0)*(z-z0)/(x1-x0)/(y1-y0)/(z1-z0) + arr_f[z_size * (vx0 * y_size + vy0) + vz1]*(x-x1)*(y-y1)*(z-z0)/(x0-x1)/(y0-y1)/(z1-z0) + arr_f[z_size * (vx0 * y_size + vy1) + vz0]*(x-x1)*(y-y0)*(z-z1)/(x0-x1)/(y1-y0)/(z0-z1) + arr_f[z_size * (vx0 * y_size + vy1) + vz1]*(x-x1)*(y-y0)*(z-z0)/(x0-x1)/(y1-y0)/(z1-z0);
  }
}

__device__ float interpolation(float *arr_f, int x_size, int y_size, int z_size, float x, float y, float z)
{
  int vx, vy, vz;
  int left_c = x;
  int right_c = left_c + 1;
  if (left_c >= FGSIZE) {
    vx = FGSIZE - 1;
  } else {
    vx = ((right_c < FGSIZE) && ((x - left_c) > 0.5)) ? right_c : left_c;
  }

  left_c = y;
  right_c = left_c + 1;
  if (left_c >= FGSIZE) {
    vy = FGSIZE - 1;
  } else {
    vy = ((right_c < FGSIZE) && ((y - left_c) > 0.5)) ? right_c : left_c;
  }

  left_c = z;
  right_c = left_c + 1;
  if (left_c >= FGSIZE) {
    vz = FGSIZE - 1;
  } else {
    vz = ((right_c < FGSIZE) && ((z - left_c) > 0.5)) ? right_c : left_c;
  }

  return arr_f[z_size * (vx * y_size + vy) + vz];
}

__global__ void kernel(float *a, float *df, bool linear_interpolation = true, bool use_texture = true)
{
  __shared__ float cache[THREADSPERBLOCK];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
	
  float x = vert[tid].x + FGSHIFT + 0.5f;
  float y = vert[tid].y + FGSHIFT + 0.5f;
  float z = vert[tid].z + FGSHIFT + 0.5f;
  
  if (use_texture) {
    cache[cacheIndex] = tex3D(df_tex, z, y, x);
  } else {
    if (linear_interpolation) {
      cache[cacheIndex] = interpolation_pl(df, FGSIZE, FGSIZE, FGSIZE, x, y, z);
    } else {
      cache[cacheIndex] = interpolation(df, FGSIZE, FGSIZE, FGSIZE, x, y, z);
    }
  }

  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (cacheIndex < s) {
	cache[cacheIndex] += cache[cacheIndex + s];
      }
      __syncthreads();
    }

  if (cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}

__global__ void kernel(float *a, Vertex *gvert, float *df, bool linear_interpolation = true, bool use_texture = true)
{
  __shared__ float cache[THREADSPERBLOCK];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
	
  float x = gvert[tid].x + FGSHIFT + 0.5f;
  float y = gvert[tid].y + FGSHIFT + 0.5f;
  float z = gvert[tid].z + FGSHIFT + 0.5f;

  if (use_texture) {
    cache[cacheIndex] = tex3D(df_tex, z, y, x);
  } else {
    if (linear_interpolation) {
      cache[cacheIndex] = interpolation_pl(df, FGSIZE, FGSIZE, FGSIZE, x, y, z);
    } else {
      cache[cacheIndex] = interpolation(df, FGSIZE, FGSIZE, FGSIZE, x, y, z);
    }
  }

  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (cacheIndex < s) {
	cache[cacheIndex] += cache[cacheIndex + s];	
      }
      __syncthreads();
    }

  if (cacheIndex == 0)
    a[blockIdx.x] = cache[0];
}	

float func(float x, float y, float z)
{
  return (0.5*sqrtf(15.0/M_PI))*(0.5*sqrtf(15.0/M_PI))*z*z*y*y*sqrtf(1.0f-z*z/RADIUS/RADIUS)/RADIUS/RADIUS/RADIUS/RADIUS;
}

void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f)
{
  for (int x = 0; x < x_size; ++x)
    for (int y = 0; y < y_size; ++y)
      for (int z = 0; z < z_size; ++z) {
	arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
      }
}

float check(Vertex *v, ptr_f f)
{
  float sum = 0.0f;
  for (int i = 0; i < VERTCOUNT; ++i)
    sum += f(v[i].x, v[i].y, v[i].z);
		
  return sum;
}

void init_vertexes()
{
  Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
  int i = 0;
  for (int iphi = 0; iphi < 2 * COEF; ++iphi)
    {	
      for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i)
	{
	  float phi = iphi * M_PI / COEF;
	  float psi = ipsi * M_PI / COEF;
	  temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
	  temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
	  temp_vert[i].z = RADIUS * cosf(psi);
	}
    }
  printf("sumcheck = %f\n", check(temp_vert, &func)*M_PI*M_PI/ COEF/COEF);
  cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) * VERTCOUNT, 0, cudaMemcpyHostToDevice);
  free(temp_vert);
}

void init_vertexes(Vertex *dev_vert)
{
  Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
  int i = 0;
  for (int iphi = 0; iphi < 2 * COEF; ++iphi)
    {	
      for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i)
	{
	  float phi = iphi * M_PI / COEF;
	  float psi = ipsi * M_PI / COEF;
	  temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
	  temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
	  temp_vert[i].z = RADIUS * cosf(psi);
	}
    }
  printf("sumcheck = %f\n", check(temp_vert, &func)*M_PI*M_PI/ COEF/COEF);
  cudaMemcpy(dev_vert, temp_vert, VERTCOUNT * sizeof(Vertex), cudaMemcpyHostToDevice);

  free(temp_vert);
}

void init_texture(float *df_h)
{
  const cudaExtent volumeSize = make_cudaExtent(FGSIZE, FGSIZE, FGSIZE);
  cudaChannelFormatDesc  channelDesc=cudaCreateChannelDesc<float>();
  cudaMalloc3DArray(&df_Array, &channelDesc, volumeSize);
  cudaMemcpy3DParms  cpyParams={0};
  cpyParams.srcPtr = make_cudaPitchedPtr( (void*)df_h, volumeSize.width*sizeof(float),  volumeSize.width,  volumeSize.height);
  cpyParams.dstArray = df_Array;
  cpyParams.extent = volumeSize;
  cpyParams.kind = cudaMemcpyHostToDevice; 
  cudaMemcpy3D(&cpyParams);
  df_tex.normalized = false;
  df_tex.filterMode = cudaFilterModeLinear;
  df_tex.addressMode[0] = cudaAddressModeClamp;
  df_tex.addressMode[1] = cudaAddressModeClamp;
  df_tex.addressMode[2] = cudaAddressModeClamp;
  cudaBindTextureToArray(df_tex, df_Array, channelDesc);
}

void release_texture()
{
  cudaUnbindTexture(df_tex); 
  cudaFreeArray(df_Array);
}

int main(int argc, char *argv[])
{
  bool use_cmemory = false, use_texture = false, linear_interpolation = false;

  char *optstr = "ctl";
  int opt = getopt(argc, argv, optstr);
  while (opt != -1) {
    switch (opt) {
    case 'c':
      use_cmemory = true;
      break;
    case 't':
      use_texture = true;
      break;
    case 'l':
      linear_interpolation = true;
      break;
    }
    opt = getopt(argc, argv, optstr);
  }
  
  Vertex *gvert;
  if (use_cmemory) {
    init_vertexes();
  } else {
    cudaMalloc((void **)&gvert, VERTCOUNT * sizeof(Vertex));
    init_vertexes(gvert);
  }

  float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
  calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);

  // for (size_t i = 0; i < FGSIZE; ++i) {
  //   printf("%f ", arr[FGSIZE * (i * FGSIZE + 1) + 1]);
  // }
  init_texture(arr);

  float *sum = (float*)malloc(sizeof(float) * BLOCKSPERGRID);
  float *sum_dev;
  cudaMalloc((void**)&sum_dev, sizeof(float) * BLOCKSPERGRID);	

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  float *darr;
  if (!use_texture) {
    cudaMalloc((void **)&darr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
    cudaMemcpy(darr, arr, sizeof(float) * FGSIZE * FGSIZE * FGSIZE, cudaMemcpyHostToDevice);
  }

  if (use_cmemory) {
    kernel<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(sum_dev, darr, linear_interpolation, use_texture);
  } else {
    kernel<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(sum_dev, gvert, darr, linear_interpolation, use_texture);
  }

  cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID, cudaMemcpyDeviceToHost);
  float s = 0.0f;
  for (int i = 0; i < BLOCKSPERGRID; ++i)
    s += sum[i];
  printf("sum = %f\n", s*M_PI*M_PI / COEF/COEF);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time: %3.1f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(sum_dev);
  if (!use_cmemory) {
    cudaFree(gvert);
  }
  if (!use_texture) {
    cudaFree(darr);
  }
  free(sum);
  release_texture();
  free(arr);

  return 0;
}
