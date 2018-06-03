/* Copyright (C) 2018 Dylan Katz, Khulan Baasan
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <err.h>

#ifdef __MACH__
#	include <mach/clock.h>
#	include <mach/mach.h>
#endif

#include "arg.h"

void usage(int);
void deviceinfo(void);
__global__ void matmulnaive(double *, double *, double *, int, int);
__global__ void matmulopt(double *, double *, double *, int, int);
void randmat(double *, int, int);
void printmat(double *, int, int);
void monotonictime(struct timespec *);
void printflops(struct timespec *, struct timespec *, int, int, int);
void testmatmul(int, int, int, bool, int, int);

char *argv0;

int
main(int argc, char *argv[])
{
	int a = 2, b = 2, c = 2, A = 20, B = 20, C = 20, tx = 8, ty = 8, i, j, k;
	bool naive = false;

	ARGBEGIN{
	case 'a':
		a = atoi(EARGF(usage(1)));
		break;
	case 'b':
		b = atoi(EARGF(usage(1)));
		break;
	case 'c':
		c = atoi(EARGF(usage(1)));
		break;
	case 'A':
		A = atoi(EARGF(usage(1)));
		break;
	case 'B':
		B = atoi(EARGF(usage(1)));
		break;
	case 'C':
		C = atoi(EARGF(usage(1)));
		break;
	case 'n':
		naive = true;
		break;
	case 's':
		tx = atoi(EARGF(usage(1)));
		break;
	case 't':
		ty = atoi(EARGF(usage(1)));
		break;
	case 'D':
		deviceinfo();
	case 'h':
		usage(0);
	default:
		usage(1);
	}ARGEND

	srand48(time(0));

	for(i = a; i < A; ++i){
		for(j = b; j < B; ++j){
			for(k = c; k < C; ++k)
				testmatmul(i, j, k, naive, tx, ty);
		}
	}
}

void
usage(int r)
{
	fprintf(r ? stderr : stdout,
		"usage: %s [-n] [-a a] [-b b] [-c c] [-A A] [-B B] [-C C] [-s s] [-t t]\n"
		"       %s [-D]\n"
		"       %s [-h]\n\n"
		" a, b, c = the minimum value of a matrix dimension (def: 2)\n"
		" A, B, C = the maximum value of a matrix dimension (def: 10)\n"
		" n       = naive MM implementation? (def: no)\n"
		" s, t    = (x, y) size of a tile (non-naive implementation) (def: 8)\n"
		" D       = show CUDA device information\n"
		" h       = show this message\n\n"
		" - The resulting matrix is A x B\n"
		" - The left hand matrix is A x C\n"
		" - The right hand matrix is C x B\n",
		argv0, argv0, argv0);

	exit(r);
}

void
deviceinfo(void)
{
	const int kb = 1024;
	const int mb = kb * kb;

	int count, i;
	cudaDeviceProp properties;

	cudaGetDeviceCount(&count);
	printf("CUDA Devices: %d\n", count);

	for(i = 0; i < count; ++i){
		cudaGetDeviceProperties(&properties, i);

		printf("\n"
			"Device %d: %s: %d.%d\n"
			"Global memory:           %zu mb\n"
			"Shared memory per block: %zu kb\n"
			"Constant memory:         %zu kb\n"
			"Registers per block:     %d\n"
			"Warp size:               %d\n"
			"Threads per block:       %d\n"
			"Maximum block size:      [%d, %d, %d]\n"
			"Maximum grid size:       [%d, %d, %d]\n"
			"Clock rate:              %d kHz\n"
			"Multiprocessor Count:    %d\n",
			i, properties.name, properties.major, properties.minor,
			properties.totalGlobalMem / mb, properties.sharedMemPerBlock / kb,
			properties.totalConstMem / kb, properties.regsPerBlock,
			properties.warpSize, properties.maxThreadsPerBlock,
			properties.maxThreadsDim[0], properties.maxThreadsDim[1],
			properties.maxThreadsDim[2], properties.maxGridSize[0],
			properties.maxGridSize[1], properties.maxGridSize[2],
			properties.clockRate, properties.multiProcessorCount);
	}

	exit(0);
}

__global__ void
matmulnaive(double *M, double *X, double *Y, int b, int c)
{
	int i = threadIdx.x / b, j = threadIdx.x % b, k;
	double r = 0.0;

	for(k = 0; k < c; ++k)
		r += Y[k*b + j] * X[i*c + k];

	M[i*b + j] = r;
}

__global__ void
matmulopt(double *M, double *X, double *Y, int b, int c)
{
	const int tx = blockDim.x, ty = blockDim.y;
	const int i = threadIdx.x + blockIdx.x*blockDim.x;
	const int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k;

	/* CUDA does not support multiple dynamically allocated shared memory
	 * arrays by default. This ugly hack works around it. Needs the third
	 * parameter of the kernel call to work correctly.
	 */
	extern __shared__ double shared[];
	double r = 0.0, *s_X = shared, *s_Y = shared + c*tx;
	/* __shared__ double s_X[c * tx], s_Y[ty * c]; */

	for(k = threadIdx.y; k < c; k += ty)
		s_X[threadIdx.x*c + k] = X[i*c + k];

	for(k = threadIdx.x; k < c; k += tx)
		s_Y[k*ty + threadIdx.y] = Y[k*b + j];

	__syncthreads();

	for(k = 0; k < c; ++k)
		r += s_Y[k*ty + threadIdx.y] * s_X[threadIdx.x*c + k];

	M[i*b + j] = r;
}

void
randmat(double *M, int a, int b)
{
	double *i;

	for(i = M; i < M + a*b; ++i)
		*i = drand48();
}

void
printmat(double *M, int a, int b)
{
	int i, j;

	for(i = 0; i < a; ++i){
		printf("[");

		for(j = 0; j < b; ++j)
			printf(" %f", M[i*b + j]);

		printf(" ]\n");
	}
}

void
monotonictime(struct timespec *ts)
{
#ifdef __MACH__
	clock_serv_t cclock;
	mach_timespec_t mts;

	host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
	clock_get_time(cclock, &mts);
	mach_port_deallocate(mach_task_self(), cclock);

	ts->tv_sec = mts.tv_sec;
	ts->tv_nsec = mts.tv_nsec;
#else
	clock_gettime(CLOCK_MONOTONIC, ts);
#endif
}

void
printflops(struct timespec *start, struct timespec *stop, int a, int b, int c)
{
	int totalops = 2 * a * b * c;
	double timediff = (stop->tv_sec - start->tv_sec) +
		1e-9*(stop->tv_nsec - start->tv_nsec);

	printf("\n"
		"Total floating-point operations = %d\n"
		"Total compute time = %f seconds\n"
		"Total flops = %f\n",
		totalops, timediff, totalops / timediff);
}

void
testmatmul(int a, int b, int c, bool naive, int tx, int ty)
{
	double *M, *X, *Y;
	double *d_M, *d_X, *d_Y;
	struct timespec start, stop, start_op, stop_op;

	/* Avoid edge cases where tile size and matrix size don't line up */
	if(!naive && (a % tx || b % ty || c % tx || c % ty))
		return;

	M = (double *)malloc(a * b * sizeof(double));
	X = (double *)malloc(a * c * sizeof(double));
	Y = (double *)malloc(c * b * sizeof(double));

	if(!M || !X || !Y)
		err(1, "malloc");

	randmat(X, a, c);
	randmat(Y, c, b);

	monotonictime(&start);

	cudaMalloc(&d_M, a * b * sizeof(double));
	cudaMalloc(&d_X, a * c * sizeof(double));
	cudaMalloc(&d_Y, c * b * sizeof(double));

	cudaMemcpy(d_X, X, a * c * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, c * b * sizeof(double), cudaMemcpyHostToDevice);

	monotonictime(&start_op);

	if(naive)
		matmulnaive<<<1, a * b>>>(d_M, d_X, d_Y, b, c);
	else
		matmulopt<<<dim3(a/tx, b/ty), dim3(tx, ty), sizeof(double)*c*(tx+ty)>>>(d_M, d_X, d_Y, b, c);

	monotonictime(&stop_op);

	cudaMemcpy(M, d_M, a * b * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_M);
	cudaFree(d_X);
	cudaFree(d_Y);

	monotonictime(&stop);

	printmat(X, a, c);
	printf(" *\n");
	printmat(Y, c, b);
	printf(" =\n");
	printmat(M, a, b);
	printflops(&start_op, &stop_op, a, b, c);
	printf("Total memory time = %f seconds\n\n",
		(stop.tv_sec-start.tv_sec) + 1e-9 * (stop.tv_nsec-start.tv_nsec));

	free(M);
	free(X);
	free(Y);
}
