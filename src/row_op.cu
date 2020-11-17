#include <cooperative_groups.h>
#include "equations.h"
#include "util.h"
#include "smem.h"
#include "debug.h"

namespace inplace {

namespace _2d {

template<typename F, typename T>
__global__ void compress_row_op(F fn, T* data, size_t batch_size, int d1, int d2) {
    T* smem = shared_memory<T>();

    size_t l = chunk_left(blockIdx.x, gridDim.x, d2);
    size_t r = chunk_right(blockIdx.x, gridDim.x, d2);
    //size_t batch_size = smem_size / (size_t)d1;
    for (size_t lv = l; lv < r; lv += batch_size) {
        batch_size = min(batch_size, r - lv);
        size_t offset = lv * (size_t)d1;
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            smem[idx] = data[offset + idx];
        }
        
        __syncthreads();
        for (size_t idx = threadIdx.x; idx < batch_size * d1; idx += blockDim.x) {
            int u = (idx / d1);
            size_t i = (lv + u) % d2;
            size_t j = idx % d1;
            fn.set_i(i);
            data[offset + idx] = smem[u * d1 + fn(j)];
        }
    }
}

template<typename F, typename T>
void compress_row_launch(F fn, T* data, int d1, int d2) {
	PRINT("Smem Compress %s\n", fn.getName().c_str());
	PRINT("\t(d1, d2) = (%d, %d)\n", d1, d2);
	size_t smem_lim = shared_mem_per_block();
	size_t smem_size = smem_lim / 32;
	int n_threads = get_num_thread(d1);
	int n_blocks = min(d2, get_num_block(compress_row_op<F, T>, n_threads, smem_size));
	PRINT("\t# blocks = %d\n", n_blocks);
	size_t batch_size = smem_size / (sizeof(T) * (size_t)d1);
	PRINT("\tbatch size = %zu\n", batch_size);
	compress_row_op<<<n_blocks, n_threads, smem_size>>>(fn, data, batch_size, d1, d2);
}

template<typename F, typename T>
__global__ void smem_row_op(F fn, T* data, int d1, int d2) {
    T* smem = shared_memory<T>();
    
    for (size_t i = blockIdx.x; i < d2; i += gridDim.x) {
        fn.set_i(i);
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
            //smem[j] = d[rm(i, s(j))];
			smem[j] = data[j + i * d1];
        }
        __syncthreads();
        for(size_t j = threadIdx.x; j < d1; j += blockDim.x) {
            //d[rm(i, j)] = smem[j];
			data[j + i * d1] = smem[fn(j)];
        }
    }
}

template<typename F, typename T>
void smem_row_launch(F fn, T* data, int d1, int d2) {
	PRINT("Smem %s\n", fn.getName().c_str());
	size_t smem_size = sizeof(T) * (size_t)d1;
	int n_threads = get_num_thread(d1);
	int n_blocks = min(d2, get_num_block(smem_row_op<F, T>, n_threads, smem_size));
	smem_row_op<<<n_blocks, n_threads, smem_size>>>(fn, data, d1, d2);
}

template<typename F, typename T>
__global__ void gmem_row_gather_op(F fn, T* data, T* tmp, int d1, int d2) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
	for (int i = 0; i < d2; i++) {
		fn.set_i(i);
		g.sync();
		for (int j = global_id; j < d1; j += grid_size) {
			//tmp[j] = d[rm(i, s(j))];
			tmp[j] = data[fn(j) + i * d1];
		}
		g.sync();
		for (int j = global_id; j < d1; j += grid_size) {
			//d[rm(i, j)] = tmp[j];
			data[j + i * d1] = tmp[j];
		}
	}
}

template<typename F, typename T>
__global__ void gmem_row_scatter_op(F fn, T* data, T* tmp, int d1, int d2) {
    namespace cg = cooperative_groups;
    cg::grid_group g = cg::this_grid();

    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int grid_size = gridDim.x * blockDim.x;
	for (int i = 0; i < d2; i++) {
		fn.set_i(i);
		g.sync();
		for (int j = global_id; j < d1; j += grid_size) {
			//tmp[j] = d[rm(i, s(j))];
			tmp[fn(j)] = data[j + i * d1];
		}
		g.sync();
		for (int j = global_id; j < d1; j += grid_size) {
			//d[rm(i, j)] = tmp[j];
			data[j + i * d1] = tmp[j];
		}
	}
}

template<typename F, typename K, typename T>
void gmem_row_launch(F fn, K kernel, T* data, int d1, int d2) {
	PRINT("Gmem %s\n", fn.getName().c_str());
	T* tmp;
	size_t tmp_size = sizeof(T) * d1;
	CudaSafeCall( cudaMallocManaged(&tmp, tmp_size) );
	prefetch(tmp, tmp_size);
	int n_threads = 1024;
	int n_blocks = get_num_block(kernel, n_threads, 0) / 2;
	PRINT("\t# blocks = %d\n", n_blocks);
	void *kernelArgs[] = {
		(void *)&fn, (void *)&data, (void *)&tmp, (void *)&d1, (void *)&d2
	};
	CudaSafeCall( cudaLaunchCooperativeKernel((void *)kernel,
										  n_blocks, n_threads, kernelArgs) );
	CudaSafeCall( cudaFree(tmp) );
}

template<typename F, typename T>
void row_gather_op(F fn, T* data, int d1, int d2) {
	size_t smem_lim = shared_mem_per_block();
	if (2 * d1 * sizeof(T) <= smem_lim / 32) {
		compress_row_launch(fn, data, d1, d2);
    }
    else if (sizeof(T) * (size_t)d1 <= smem_lim) {
		smem_row_launch(fn, data, d1, d2);
    }
	else {
        gmem_row_launch(fn, gmem_row_gather_op<F, T>, data, d1, d2);
    }
}

template<typename F, typename T>
void row_scatter_op(F fn, T* data, int d1, int d2) {
	gmem_row_launch(fn, gmem_row_scatter_op<F, T>, data, d1, d2);
}

template void row_gather_op(c2r::row_shuffle, float*, int, int);
template void row_gather_op(c2r::row_shuffle, double*, int, int);

template void row_gather_op(r2c::row_shuffle, float*, int, int);
template void row_gather_op(r2c::row_shuffle, double*, int, int);

template void row_gather_op(_3d::_321::row_shuffle, float*, int, int);
template void row_gather_op(_3d::_321::row_shuffle, double*, int, int);

template void row_scatter_op(r2c::row_scatter_shuffle, float*, int, int);
template void row_scatter_op(r2c::row_scatter_shuffle, double*, int, int);

template void row_scatter_op(_3d::_321::row_scatter_shuffle, float*, int, int);
template void row_scatter_op(_3d::_321::row_scatter_shuffle, double*, int, int);

}

}
