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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <err.h>

#include "arg.h"

void usage(int);
void matmul(double *, double *, double *, int, int, int);
void printmul(double *, int, int);
void testmul(int, int, int);

char *argv0;

int
main(int argc, char *argv[])
{
	int a = 2, b = 2, c = 2, A = 10, B = 10, C = 10;

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

	for(; a < A; ++a){
		for(; b < B; ++b){
			for(; c < C; ++c)
				testmul(a, b, c);
		}
	}
}

void
usage(int r)
{
	fprintf(r ? stderr : stdout,
		"usage: %s [-h] [-a a] [-b b] [-c c] [-A A] [-B B] [-C C]\n"
		" a, b, c = the minimum value of the iterations (def: 2)\n"
		" A, B, C = the maximum value of the iterations (def: 10)\n",
		argv0);
	exit(r);
}

void
matmul(double *C, double *A, double *B, int a, int b, int c)
{
	int i, j, k;

	for(i = 0; i < a; ++i){
		for(j = 0; j < b; ++j){
			C[i*b + j] = 0;

			for(k = 0; k < c; ++k)
				C[i*b + j] += B[k*b + j] * A[i*c + k];
		}
	}
}

void
printmul(double *M, int a, int b)
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
testmul(int a, int b, int c)
{
	double *A, *B, *C, *i;

	C = malloc(a * b * sizeof(double));
	A = malloc(a * c * sizeof(double));
	B = malloc(c * b * sizeof(double));

	if(!C || !A || !B)
		err(1, "calloc");

	for(i = A; i < A + a*c; ++i)
		*i = drand48();

	for(i = B; i < B + c*b; ++i)
		*i = drand48();

	matmul(C, A, B, a, b, c);

	printmul(A, a, c);
	printf("*\n");
	printmul(B, c, b);
	printf("=\n");
	printmul(C, a, b);
	printf("~~~~~~~~~~~~~~~~~~~~~\n");

	free(A);
	free(B);
	free(C);
}
