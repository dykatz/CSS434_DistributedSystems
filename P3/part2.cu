/* Copyright (C) 2018 Dylan Katz
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

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <cstdio>

#include "arg.h"

char *argv0;

[[noreturn]] void usage(void);
[[noreturn]] void dump_device_data(void);
void random_inputs(float *, float *, int);
__global__ void add_by_block(float *, float *, float *);
__global__ void add_by_thread(float *, float *, float *);

int
main(int argc, char *argv[])
{
	int N;
	bool t = false, m = false;

	ARGBEGIN {
	case 'n': {
		std::istringstream iss(EARGF(usage()));

		if (!(iss >> N))
			usage();
	} break;
	case 't': {
		t = true;
	} break;
	case 'm': {
		m = true;
	} break;
	case 'D':
		dump_device_data();
	case 'V':
		std::cout << "v0.0.1" << std::endl;
		std::exit(0);
	default:
		usage();
	} ARGEND;

	if (t && m)
		usage();

	float *ha, *hb, *hc, *da, *db, *dc;

	ha = (float *)malloc(N * sizeof(float));
	hb = (float *)malloc(N * sizeof(float));
	hc = (float *)malloc(N * sizeof(float));

	random_inputs(ha, hb, N);

	auto start = std::chrono::system_clock::now();

	cudaMalloc((void **)&da, N * sizeof(float));
	cudaMalloc((void **)&db, N * sizeof(float));
	cudaMalloc((void **)&dc, N * sizeof(float));

	cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

	auto post_alloc = std::chrono::system_clock::now();

	if (m) {
		// XXX
	} else if (t) {
		add_by_thread<<<1, N>>>(da, db, dc);
	} else {
		add_by_block<<<N, 1>>>(da, db, dc);
	}

	cudaDeviceSynchronize();

	auto post_op = std::chrono::system_clock::now();

	cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < N; ++i) {
		std::printf("[ %7.3f ] + [ %7.3f ] = [ %8.3f ]\n",
			ha[i], hb[i], hc[i]);
	}

	free(ha);
	free(hb);
	free(hc);

	std::chrono::duration<double> total = end - start;
	std::chrono::duration<double> memalloc0 = post_alloc - start;
	std::chrono::duration<double> memalloc1 = end - post_op;
	std::chrono::duration<double> operation = post_op - post_alloc;

	std::cout << "Total elapsed time = " << total.count() << "s" << std::endl;
	std::cout << "Operation time = " << operation.count() << "s" << std::endl;
	std::cout << "GPU memory allocation time = " << memalloc0.count() << "s" << std::endl;
	std::cout << "GPU memory free time = " << memalloc1.count() << "s" << std::endl;
	return 0;
}

void
usage(void)
{
	std::cout << "usage: " << argv0 << " [-n size:int] [-m | -t]" << std::endl;
	std::cout << "       " << argv0 << " [-D]" << std::endl;
	std::cout << "       " << argv0 << " [-V]" << std::endl;
	std::exit(1);
}

void
dump_device_data(void)
{
	const int kb = 1024;
	const int mb = kb * kb;

	int device_count;
	cudaGetDeviceCount(&device_count);
	std::cout << "CUDA Devices: " << device_count << std::endl;

	for (int i = 0; i < device_count; ++i) {
		cudaDeviceProp p;
		cudaGetDeviceProperties(&p, i);

		std::cout << std::endl << "Device " << i << ": " << p.name;
		std::cout << ": " << p.major << "." << p.minor << std::endl;

		std::cout << "Global memory: " << (p.totalGlobalMem / mb);
		std::cout << "mb" << std::endl;

		std::cout << "Shared memory: " << (p.sharedMemPerBlock / kb);
		std::cout << "kb" << std::endl;

		std::cout << "Const memory: " << (p.totalConstMem / kb);
		std::cout << "kb" << std::endl;

		std::cout << "Block regs: " << p.regsPerBlock << std::endl;
		std::cout << "Warp size: " << p.warpSize << std::endl;
		std::cout << "Threads/block: " << p.maxThreadsPerBlock << std::endl;

		std::cout << "Max block size: [" << p.maxThreadsDim[0] << ", ";
		std::cout << p.maxThreadsDim[1] << ", " << p.maxThreadsDim[2];
		std::cout << "]" << std::endl;

		std::cout << "Max grid size: [" << p.maxGridSize[0] << ", ";
		std::cout << p.maxGridSize[1] << ", " << p.maxGridSize[2];
		std::cout << "]" << std::endl;
	}

	std::exit(0);
}

void
random_inputs(float *a, float *b, int n)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(1, 1000);

	for (int i = 0; i < n; ++i) {
		a[i] = dist(gen);
		b[i] = dist(gen);
	}
}

__global__ void
add_by_block(float *a, float *b, float *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void
add_by_thread(float *a, float *b, float *c)
{
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
