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
void testmatmul(int, int, int, bool, int, int);

char *argv0;

int
main(int argc, char *argv[])
{
	int a = 2, b = 2, c = 2, A = 10, B = 10, C = 10, tx = 8, ty = 8, i, j, k;
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
			"Global memory:     %zumb\n"
			"Shared memory:     %zukb\n"
			"Const memory:      %zukb\n"
			"Block registers:   %d\n"
			"Warp size:         %d\n"
			"Threads per block: %d\n"
			"Max block size:    [%d, %d, %d]\n"
			"Max grid size:     [%d, %d, %d]\n",
			i, properties.name, properties.major, properties.minor,
			properties.totalGlobalMem / mb, properties.sharedMemPerBlock / kb,
			properties.totalConstMem / kb, properties.regsPerBlock,
			properties.warpSize, properties.maxThreadsPerBlock,
			properties.maxThreadsDim[0], properties.maxThreadsDim[1],
			properties.maxThreadsDim[2], properties.maxGridSize[0],
			properties.maxGridSize[1], properties.maxGridSize[2]);
	}

	exit(0);
}

__global__ void
matmulnaive(double *M, double *X, double *Y, int b, int c)
{
	int i = threadIdx.x / b;
	int j = threadIdx.x % b;
	int k;

	M[i*b + j] = 0;

	for(k = 0; k < c; ++k)
		M[i*b + j] += Y[k*b + j] * X[i*c + k];
}

__global__ void
matmulopt(double *M, double *X, double *Y, int b, int c)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k;

	M[i*b + j] = 0;

	for(k = 0; k < c; ++k)
		M[i*b + j] += Y[k*b + j] * X[i*c + k];
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
testmatmul(int a, int b, int c, bool naive, int tx, int ty)
{
	double *M, *X, *Y;
	double *d_M, *d_X, *d_Y;
	struct timespec start, stop;

	if(!naive && (a % tx || b % ty))
		return;

	M = (double *)malloc(a * b * sizeof(double));
	X = (double *)malloc(a * c * sizeof(double));
	Y = (double *)malloc(c * b * sizeof(double));

	if(!M || !X || !Y)
		err(1, "malloc");

	randmat(X, a, c);
	randmat(Y, c, b);

	cudaMalloc(&d_M, a * b * sizeof(double));
	cudaMalloc(&d_X, a * c * sizeof(double));
	cudaMalloc(&d_Y, c * b * sizeof(double));

	monotonictime(&start);

	cudaMemcpy(d_X, X, a * c * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y, Y, c * b * sizeof(double), cudaMemcpyHostToDevice);

	if(naive)
		matmulnaive<<<1, a * b>>>(M, X, Y, b, c);
	else
		matmulopt<<<dim3(a/tx, b/ty), dim3(tx, ty)>>>(M, X, Y, b, c);

	cudaMemcpy(M, d_M, a * b * sizeof(double), cudaMemcpyDeviceToHost);

	monotonictime(&stop);

	cudaFree(d_M);
	cudaFree(d_X);
	cudaFree(d_Y);

	printmat(X, a, c);
	printf("*\n");
	printmat(Y, c, b);
	printf("=\n");
	printmat(M, a, b);
	printf("This operation took %f seconds\n~~~~~~~~~~~~~~~~~~~~~\n",
		(stop.tv_sec-start.tv_sec) + 1e-9*(stop.tv_nsec-start.tv_nsec));

	free(M);
	free(X);
	free(Y);
}
