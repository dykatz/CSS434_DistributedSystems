#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <err.h>

/* C =  A * B
 * A : [c x b]
 * B : [a x c]
 * C : [a x b]
 */
void matmul(double *, double *, double *, int, int, int);
void printmul(double *, int, int);
void testmul(int, int, int);

int
main(int argc, char *argv[])
{
	int a, b, c;

	srand48(time(0));

	for(a = 2; a < 10; ++a)
		for(b = 2; b < 10; ++b)
			for(c = 2; c < 10; ++c)
				testmul(a, b, c);
}

void
matmul(double *C, double *A, double *B, int a, int b, int c)
{
	int i, j, k;

	for(i = 0; i < a; ++i){
		for(j = 0; j < b; ++j){
			C[i*b + j] = 0;

			for(k = 0; k < c; ++k)
				C[i*b + j] = B[k*b + j] * A[i*c + k];
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
	double *A, *B, *C;
	int i;

	C = calloc(a * b, sizeof(double));
	A = calloc(a * c, sizeof(double));
	B = calloc(c * b, sizeof(double));

	if (!C || !A || !B)
		errx(1, "calloc");

	for(i = 0; i < a * c; ++i)
		A[i] = drand48();

	for(i = 0; i < c * b; ++i)
		B[i] = drand48();

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
