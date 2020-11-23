#include <cooperative_groups.h>
#include "util.h"
#include "equations.h"
#include "smem.h"
#include "debug.h"

namespace inplace {

namespace _3d {

template<typename F, typename T>
__global__ void smem_col_op(F fn, T* data, int d1, int d2, int d3) {
	T* smem = shared_memory<T>();
	
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				smem[i * blockDim.x + threadIdx.x] = data[j + i * d1 + kd1d2];
			}
			__syncthreads();
			for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
				size_t i_prime = fn(i, j);
				data[j + i * d1 + kd1d2] = smem[i_prime * blockDim.x + threadIdx.x];
			}
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int d1, int d2, int d3) {
	PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2(x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, d1, d2, d3);
}

template<typename F, typename T>
__global__ void gmem_col_op(F fn, T* data, T* tmp, int d1, int d2, int d3) {
	namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
	
	for (size_t k = 0; k < d3; k++) {
		size_t kd1d2 = k * d1 * d2;
		for (size_t j = threadIdx.x; j < d1; j += blockDim.x) {
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				tmp[threadIdx.x + i * blockDim.x] = data[j + i * d1 + kd1d2];
			}
			g.sync();
			for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
				size_t i_prim = fn(i, j);
				data[j + i * d1 + kd1d2] = tmp[threadIdx.x + i_prim * blockDim.x];
			}
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int d1, int d2, int d3) {
	PRINT("Gmem %s\n", fn.getName().c_str());
	int n_threads = 256;
	_2d::c2r::rotate r1;
	_2d::r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&tmp, (void *)&d1, (void *)&d2, (void *)&d3
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int d1, int d2, int d3) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, d1, d2, d3);
	}
	else {
		gmem_launch(fn, data, d1, d2, d3);
	}
}

template void col_op(_2d::c2r::rotate, float*, int, int, int);
template void col_op(_2d::c2r::rotate, double*, int, int, int);
template void col_op(_2d::c2r::col_shuffle, float*, int, int, int);
template void col_op(_2d::c2r::col_shuffle, double*, int, int, int);

template void col_op(_2d::r2c::rotate, float*, int, int, int);
template void col_op(_2d::r2c::rotate, double*, int, int, int);
template void col_op(_2d::r2c::col_shuffle, float*, int, int, int);
template void col_op(_2d::r2c::col_shuffle, double*, int, int, int);

} //End of namespace _3d

namespace _2d {

template<typename F, typename T>
__global__ void smem_col_op(F fn, T* data, int d1, int d2) {
	T* smem = shared_memory<T>();
	
	for (size_t j = threadIdx.x + blockIdx.x * blockDim.x; j < d1; j += gridDim.x * blockDim.x) {
		__syncthreads();
		for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
			smem[i * blockDim.x + threadIdx.x] = data[j + i * d1];
		}
		__syncthreads();
		for (size_t i = threadIdx.y; i < d2; i += blockDim.y) {
			size_t i_prime = fn(i, j);
			data[j + i * d1] = smem[i_prime * blockDim.x + threadIdx.x];
		}
	}
}

template<typename F, typename T>
void smem_launch(F fn, T* data, int d1, int d2) {
	PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_lim = shared_mem_per_block();
	int x_lim = smem_lim / (sizeof(T) * d2);
	int n_threads_x = min(1024, (int)pow(2, (int)log2(x_lim)));
	int n_threads_y = min(d2, 1024 / n_threads_x);
	dim3 block_dim(n_threads_x, n_threads_y);
	int n_threads = n_threads_x * n_threads_y;
	size_t smem_size = sizeof(T) * d2 * n_threads_x;
	int n_blocks = min(div_up(d1, n_threads_x), get_num_block(smem_col_op<F, T>, n_threads, smem_size));
	smem_col_op<<<n_blocks, block_dim, smem_size>>>(fn, data, d1, d2);
}

template<typename F, typename T>
__global__ void gmem_col_op(F fn, T* data, T* tmp, int d1, int d2) {
	namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();
	
	for (size_t j = threadIdx.x; j < d1; j += blockDim.x) {
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			tmp[threadIdx.x + i * blockDim.x] = data[j + i * d1];
		}
		g.sync();
		for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
			size_t i_prim = fn(i, j);
			data[j + i * d1] = tmp[threadIdx.x + i_prim * blockDim.x];
		}
	}
}

template<typename F, typename T>
void gmem_launch(F fn, T* data, int d1, int d2) {
	PRINT("Gmem %s\n", fn.getName().c_str());
	int n_threads = 256;
	c2r::rotate r1;
	r2c::rotate r2;
	if (typeid(fn) != typeid(r1) && typeid(fn) != typeid(r2)) {
		const int upper_lim = 18;
		if (msb(d1) <= upper_lim) {
			n_threads = 32;
		}
	}
	int n_blocks = min(d2, get_num_block(gmem_col_op<F, T>, n_threads, 0)) / 2;
	PRINT("\t# blocks = %d\n", n_blocks);
	T* tmp;
	size_t tmp_size = sizeof(T) * n_threads * d2;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&tmp, (void *)&d1, (void *)&d2
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)gmem_col_op<F, T>,
				  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void col_op(F fn, T* data, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (smem_lim / (sizeof(T) * d2) >= 16) {
		smem_launch(fn, data, d1, d2);
	}
	else {
		gmem_launch(fn, data, d1, d2);
	}
}

template void col_op(c2r::rotate, float*, int, int);
template void col_op(c2r::rotate, double*, int, int);
template void col_op(c2r::col_shuffle, float*, int, int);
template void col_op(c2r::col_shuffle, double*, int, int);

template void col_op(r2c::rotate, float*, int, int);
template void col_op(r2c::rotate, double*, int, int);
template void col_op(r2c::col_shuffle, float*, int, int);
template void col_op(r2c::col_shuffle, double*, int, int);
}
}
