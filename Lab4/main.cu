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

[[noreturn]] void usage(const char **, const char *);
void random_inputs(float *, float *, int);
__global__ void add(float *, float *, float *);

int
main(int argc, const char **argv)
{
	int N;

	if (argc == 1) {
		N = 512;
	} else if (argc == 2) {
		std::istringstream iss(argv[1]);

		if (!(iss >> N))
			usage(argv, "first argument must be a number");
	} else {
		usage(argv, "gave more than one argument");
	}

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

	add<<<N, 1>>>(da, db, dc);
	cudaDeviceSynchronize();

	cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < N; ++i) {
		std::cout << "[ " << ha[i] << " ] + [ " << hb[i];
		std::cout << " ] = [ " << hc[i] << " ]" << std::endl;
	}

	free(ha);
	free(hb);
	free(hc);

	std::chrono::duration<double> dt = end - start;
	std::cout << "Elapsed time = " << dt.count() << "s" << std::endl;
	return 0;
}

[[noreturn]] void
usage(const char **argv, const char *errmsg)
{
	if (errmsg != NULL)
		std::cerr << "err: " << errmsg << std::endl;

	std::cerr << "usage: " << argv[0] << " [size]" << std::endl;
	std::exit(1);
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
add(float *a, float *b, float *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
