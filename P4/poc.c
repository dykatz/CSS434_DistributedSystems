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
void matmul(double *, double *, double *, int, int, int);
void randmat(double *, int, int);
void printmat(double *, int, int);
void monotonictime(struct timespec *);
void printflops(struct timespec *, struct timespec *, int, int, int);
void testmatmul(int, int, int);

char *argv0;

int
main(int argc, char *argv[])
{
	int a = 2, b = 2, c = 2, A = 10, B = 10, C = 10, i, j, k;

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
	case 'h':
		usage(0);
	default:
		usage(1);
	}ARGEND

	srand48(time(0));

	for(i = a; i < A; ++i){
		for(j = b; j < B; ++j){
			for(k = c; k < C; ++k)
				testmatmul(i, j, k);
		}
	}
}
void
usage(int r)
{
	fprintf(r ? stderr : stdout,
		"usage: %s [-a a] [-b b] [-c c] [-A A] [-B B] [-C C]\n"
		"       %s [-h]\n\n"
		" a, b, c = the minimum value of a matrix dimension (def: 2)\n"
		" A, B, C = the maximum value of a matrix dimension (def: 10)\n"
		" h       = show this message\n\n"
		" - The resulting matrix is A x B\n"
		" - The left hand matrix is A x C\n"
		" - The right hand matrix is C x B\n",
		argv0, argv0);
	exit(r);
}

void
matmul(double *M, double *X, double *Y, int a, int b, int c)
{
	int i, j, k;

	for(i = 0; i < a; ++i){
		for(j = 0; j < b; ++j){
			M[i*b + j] = 0;

			for(k = 0; k < c; ++k)
				M[i*b + j] += Y[k*b + j] * X[i*c + k];
		}
	}
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
		"Total flops = %f\n\n",
		totalops, timediff, totalops / timediff);
}

void
testmatmul(int a, int b, int c)
{
	double *M, *X, *Y;
	struct timespec start, stop;

	M = malloc(a * b * sizeof(double));
	X = malloc(a * c * sizeof(double));
	Y = malloc(c * b * sizeof(double));

	if(!M || !X || !Y)
		err(1, "malloc");

	randmat(X, a, c);
	randmat(Y, c, b);

	monotonictime(&start);
	matmul(M, X, Y, a, b, c);
	monotonictime(&stop);

	printmat(X, a, c);
	printf(" *\n");
	printmat(Y, c, b);
	printf(" =\n");
	printmat(M, a, b);
	printflops(&start, &stop, a, b, c);

	free(M);
	free(X);
	free(Y);
}
