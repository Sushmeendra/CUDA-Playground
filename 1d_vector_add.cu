#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>


const int N = 16384;
const int THREADS_PER_BLOCK = 512;
const int NUM_BLOCKS = (N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK;

/* Running one thread in each block */
__global__ void add_blocks (int *a, int *b, int *c)
{
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

/* Running multiple threads in one block */
__global__ void add_threads (int *a, int *b, int *c)
{
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}


__global__ void add_threads_blocks (int *a, int *b, int *c, int n) {
  
  int index = threadIdx.x * blockIdx.x * threadIdx.x;

  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

int main(void) 
{
  int *a, *b, *c; /* Host (CPU) copies of a, b, c */
  int *d_a, *d_b, *d_c; /* Device (GPU) copies of a, b, c */
  size_t size = N * sizeof(int);


  /* Allocate memory in device */
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);

  /* Allocate memory in host */
  a = (int *) malloc(size);
  b = (int *) malloc(size);
  c = (int *) malloc(size);

  /* Allocate random data in vectors a and b (inside host) */
  for (int i = 0; i < N; ++i) 
  {
    a[i] = rand();
    b[i] = rand();
  }

  /* Copy data to device (GPU) */
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  /* Launching add() kernel on device with N blocks and 1 thread */
  add_blocks<<<N,1>>>(d_a, d_b, d_c);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  /* Sanity Check */
  for (int i = 0; i < N; ++i) {
    assert(c[i] == a[i] + b[i]);
  }
  printf("Version with %d blocks executed succesfully!\n", N);

  /* Launching add() kernel on device with 1 block and N threads */
  add_threads<<<1,N>>>(d_a, d_b, d_c);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  /* Sanity Check */
  for (int i = 0; i < N; ++i) {
    assert(c[i] == a[i] + b[i]);
  }
  printf("Version with %d threads executed succesfully!\n", N);

  /* Launching add() kernel on device with N threads and NUM_BLOCKS blocks */
  add_threads_blocks<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  /* Sanity Check */
  for (int i = 0; i < N; ++i) {
    assert(c[i] == a[i] + b[i]);
  }
  printf("Version with %d threads/blocks executed succesfully!\n", N);

  /* Clean-up */
  free(a); free(b); free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  return 0;
}
