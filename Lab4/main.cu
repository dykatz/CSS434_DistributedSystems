#include <chrono>
#include <iostream>
#include <random>

void random_inputs(float *, float *, int);
__global__ void add(float *, float *, float *);

int
main(int argc, char *argv[])
{
	int N = 32;
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

	cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < N; ++i) {
		std::cout << "[ " << a[i] << " ] + [ " << b[i];
		std::cout << " ] = [ " << c[i] << " ]" << std::endl;
	}

	free(ha);
	free(hb);
	free(hc);

	std::chrono::duration<double> dt = end - start;
	std::cout << "Elapsed time = " << dt.count() << "s" << std::endl;
	return 0;
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
