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
void random_inputs(float *, float *, int);

int
main(int argc, char *argv[])
{
	int N = 512;

	ARGBEGIN {
	case 'n': {
		std::istringstream iss(EARGF(usage()));

		if (!(iss >> N))
			usage();
	} break;
	case 'V':
		std::cout << "v0.0.1" << std::endl;
		std::exit(0);
	default:
		usage();
	} ARGEND;

	float *ha, *hb, *hc;

	ha = (float *)malloc(N * sizeof(float));
	hb = (float *)malloc(N * sizeof(float));
	hc = (float *)malloc(N * sizeof(float));

	random_inputs(ha, hb, N);

	auto start = std::chrono::system_clock::now();

	for (int i = 0; i < N; ++i)
		hc[i] = ha[i] + hb[i];

	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < N; ++i) {
		std::printf("[ %7.3f ] + [ %7.3f ] = [ %8.3f ]\n",
			ha[i], hb[i], hc[i]);
	}

	free(ha);
	free(hb);
	free(hc);

	std::chrono::duration<double> dt = end - start;
	std::cout << "Elapsed time = " << dt.count() << "s" << std::endl;
	return 0;
}

void
usage(void)
{
	std::cout << "usage: " << argv0 << " [-n size:int]" << std::endl;
	std::cout << "       " << argv0 << " [-V]" << std::endl;
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
