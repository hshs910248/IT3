#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include "transpose.h"
#include "tensor_util.h"
#include "cudacheck.h"

template<typename T>
void transpose(TensorUtil<T>& tu) {
	size_t& vol = tu.vol;
	int perm_int = 100 * (tu.permutation[0] + 1) + 10 * (tu.permutation[1] + 1) + (tu.permutation[2] + 1);
	printf("Inplace %dtranspose\n", perm_int);
	printf("Dims = (%zu %zu %zu)\n", tu.dim_long[0], tu.dim_long[1], tu.dim_long[2]);
	//printf("Vol = %zu\n", tu.vol);

	T* d_data = NULL;
	size_t dataSize = vol * sizeof(T);
    printf("Data size = %.5f GB\n", (double)dataSize / 1e9);
	CudaSafeCall( cudaMallocManaged(&d_data, dataSize) );
	tu.init_data(d_data);
	//if (tu.fp == NULL) tu.print_tensor(d_data);
	
	cudaEvent_t start, stop;
	CudaSafeCall( cudaEventCreate(&start) );
	CudaSafeCall( cudaEventCreate(&stop) );
	CudaSafeCall( cudaEventRecord(start, 0) );
	
	inplace::transpose(d_data, tu.rank, tu.dim, tu.permutation, sizeof(T));
    
	CudaSafeCall( cudaDeviceSynchronize() );
	CudaSafeCall( cudaEventRecord(stop, 0) );
	CudaSafeCall( cudaEventSynchronize(stop) );
	float t;
	CudaSafeCall( cudaEventElapsedTime(&t, start, stop) );
	float throughput = ((double)dataSize * sizeof(T) * 2) / 1e6 / t;
	printf("Execution Time: %.5fms\nEffective Bandwidth: %.5fGB/s\n", t, throughput);
    FILE* txtfp = fopen("inplace_bench.txt", "a+");
    fprintf(txtfp, "%.5f\n", t);
    fclose(txtfp);
	txtfp = fopen("inplace_bench_throughput.txt", "a+");
    fprintf(txtfp, "%.5f\n", throughput);
    fclose(txtfp);

    //if (tu.fp != NULL) tu.write_file(d_data);
	//else tu.print_tensor(d_data);
	
	CudaSafeCall( cudaFree(d_data) );
}

int main(int argc, char** argv) {
	/*if (argc != 8 && argc != 9) {
		printf("Usage: [d1][d2][d3][Permutation][Size of data type in byte][Output file]\nOutput file could be ignored\n");
		return 0;
	}*/
	int rank = 3;
	int k = 1;
	
	int dim[3];
	for (int i = 0; i < rank; i++) {
		dim[i] = atoi(argv[k++]);
	}
	
	int permutation[3];
	for (int i = 0; i < rank; i++) {
		permutation[i] = atoi(argv[k++]);
	}
	int type_size = atoi(argv[k++]);
	FILE* fp = NULL;
	if (argc == k + 1) {
		fp = fopen(argv[k], "wb");
	}
	if (type_size == 4) {
		TensorUtil<int> tu(fp, rank, dim, permutation);
		transpose<int>(tu);
	}
	else {
		TensorUtil<long long> tu(fp, rank, dim, permutation);
		transpose<long long>(tu);
	}
	if (fp != NULL) fclose(fp);
	
	return 0;
}
